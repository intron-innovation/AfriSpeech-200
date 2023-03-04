###### Code adapted from  ######
# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2
# https://github.com/openai/whisper
################################

import os

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/'
os.environ['XDG_CACHE_HOME'] = '/data/.cache/'

import gc
import numpy as np
import torch
import time
import pandas as pd
import whisper
import jiwer
from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoModelForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_afri_speech_data
from src.utils.text_processing import clean_text
from src.utils.utils import parse_argument, write_pred_inference_df

gc.collect()
torch.cuda.empty_cache()

processor = None
device = None


class AfriSpeechWhisperDataset(torch.utils.data.Dataset):
    """
    A simple class to wrap AfriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, data_path, split="dev", device="cpu", model_id="whisper",
                 max_audio_len_secs=17, audio_dir="./data/", gpu=-1
                 ):
        self.dataset = load_afri_speech_data(
            data_path=data_path,
            max_audio_len_secs=max_audio_len_secs,
            audio_dir=audio_dir,
            split=split, gpu=gpu
        )
        self.device = device
        self.model_id = model_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio_path = self.dataset[item]['audio_paths']
        text = self.dataset[item]['text']
        accent = self.dataset[item]['accent']

        audio = load_audio_file(audio_path)
        if 'whisper' in self.model_id and os.path.isdir(args.model_id_or_path):
            input_features = processor(
                audio,
                sampling_rate=AudioConfig.sr, 
                return_tensors="pt",
            )
            audio = input_features.input_features.squeeze()
        elif 'whisper' in self.model_id:
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = processor(
                audio, sampling_rate=AudioConfig.sr, padding='max_length', 
                max_length=AudioConfig.sr*17, truncation=True
            )
            audio = input_features.input_values[0]

        return (audio, text, audio_path, accent)

    
    

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split="test", device="cpu", model_id="whisper",
                 max_audio_len_secs=17, gpu=-1
                 ):
        self.dataset = load_dataset("librispeech_asr", "clean", split=split)
        self.device = device
        self.model_id = model_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio = self.dataset[item]['audio']['array']
        text = self.dataset[item]['text']
        accent = "US English"
        audio_path = self.dataset[item]['file']
        
        audio = np.asarray(audio)
        if 'whisper' in self.model_id and os.path.isdir(self.model_id):
            input_features = processor(
                audio,
                sampling_rate=AudioConfig.sr, 
                return_tensors="pt",
            )
            audio = input_features.input_features#.squeeze()
        elif 'whisper' in self.model_id:
            audio = np.asarray(audio, dtype=np.float32)
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = processor(
                audio, sampling_rate=AudioConfig.sr, padding='max_length', 
                max_length=AudioConfig.sr*17, truncation=True
            )
            audio = input_features.input_values[0]

        return (audio, text, audio_path, accent)

    
def transcribe_whisper(args, model, loader, split):
    tsince = int(round(time.time()))
    hypotheses = []
    references = []
    paths = []
    accents = []
    #if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
    #    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
    options = whisper.DecodingOptions(language="en", fp16=args.gpu > -1,
                                      without_timestamps=True)

    # options = dict(language=language, beam_size=5, best_of=5)
    # transcribe_options = dict(task="transcribe", **options)
    # transcription = model.transcribe(audio, **transcribe_options)["text"]

    for audio_or_mels, texts, audio_path, accent in tqdm(loader):
        if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
            audio_or_mels = audio_or_mels.to(device)
            with torch.no_grad():
                pred_ids = model.generate(audio_or_mels)
            results = processor.batch_decode(pred_ids, skip_special_tokens = True)
            hypotheses.extend([result for result in results])
        elif 'whisper' in args.model_id_or_path:
            results = model.decode(audio_or_mels, options)
            hypotheses.extend([result.text for result in results])
        else:
            audio_or_mels = audio_or_mels.to(device)
            with torch.no_grad():
                logits = model(audio_or_mels).logits
            pred_ids = torch.argmax(torch.tensor(logits), dim=-1)
            results = processor.batch_decode(pred_ids)
            hypotheses.extend([result for result in results])
        
        references.extend(texts)
        paths.extend(audio_path)
        accents.extend(accent)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references,
                             audio_paths=paths, accent=accents))

    pred_clean = [clean_text(text) for text in data["hypothesis"]]
    ref_clean = [clean_text(text) for text in data["reference"]]
    
    pred_clean = [text if text != "" else "abcxyz" for text in pred_clean]
    ref_clean = [text if text != "" else "abcxyz" for text in ref_clean]
    
    data["pred_clean"] = pred_clean
    data["ref_clean"] = ref_clean

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    print(f"Cleanup WER: {all_wer * 100:.2f} %")

    normalizer = EnglishTextNormalizer()
    
    pred_normalized = [normalizer(text) for text in data["hypothesis"]]
    gt_normalized = [normalizer(text) for text in data["reference"]]
    
    pred_normalized = [text if text != "" else "abcxyz" for text in pred_normalized]
    gt_normalized = [text if text != "" else "abcxyz" for text in gt_normalized]

    whisper_wer = jiwer.wer(gt_normalized, pred_normalized)
    print(f"EnglishTextNormalizer WER: {whisper_wer * 100:.2f} %")

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    
    write_pred_inference_df(args.model_id_or_path, data, all_wer, split=split)
    
    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{args.model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / len(data):.4f}s per sample"
    )


if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu")
    print(device)
    
    if "librispeech" in args.data_csv_path:
        dataset = LibriSpeechDataset(data_path="librispeech_asr", split='test',
                                    model_id=args.model_id_or_path,
                                    device=device,)
        split = 'test-libri-speech'
        
    else:
        split = args.data_csv_path.split("-")[1]
        dataset = AfriSpeechWhisperDataset(data_path=args.data_csv_path,
                                           max_audio_len_secs=args.max_audio_len_secs,
                                           audio_dir=args.audio_dir, device=device,
                                           split=split,
                                           gpu=args.gpu, model_id=args.model_id_or_path
                                           )
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize)
    
    if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
        # load model and processor
        try:
            processor = WhisperProcessor.from_pretrained(args.model_id_or_path)
        except Exception as e:
            processor = WhisperProcessor.from_pretrained(os.path.dirname(args.model_id_or_path))
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id_or_path)
    elif "whisper" in args.model_id_or_path:
        whisper_model = args.model_id_or_path.split("_")[1]
        model = whisper.load_model(whisper_model)
        print(
            f"Model {whisper_model} is {'multilingual' if model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )
    else:
        processor = Wav2Vec2Processor.from_pretrained(args.model_id_or_path)
        model = AutoModelForCTC.from_pretrained(args.model_id_or_path).to(device)

    model = model.to(device)

    transcribe_whisper(args, model, data_loader, split)
