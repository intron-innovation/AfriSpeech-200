###### Code adapted from  ######
# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2
# https://github.com/openai/whisper
################################

import os
data_home = "data"
os.environ['TRANSFORMERS_CACHE'] = f'/{data_home}/.cache/'
os.environ['XDG_CACHE_HOME'] = f'/{data_home}/.cache/'

import numpy as np
import torch
import time
import pandas as pd
import whisper
import jiwer
import json 
from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
import evaluate
from transformers import Wav2Vec2Processor, AutoModelForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_afri_speech_data, DISCRIMINATIVE
from src.utils.text_processing import clean_text, strip_task_tags, get_task_tags
from src.utils.utils import parse_argument, write_pred_inference_df
from src.train.models import Wav2Vec2ForCTCnCLS

processor = None
device = None
metric = evaluate.load("f1")


class AfriSpeechWhisperDataset(torch.utils.data.Dataset):
    """
    A simple class to wrap AfriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, data_path, split="dev", device="cpu", model_id="whisper",
                 max_audio_len_secs=17, audio_dir=f"./{data_home}/", gpu=-1
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
        with open(args.model_id_or_path+'/label_map.json') as file:
            LABEL_MAP = json.load(file)
        audio_path = self.dataset[item]['audio_paths']
        text = self.dataset[item]['text']
        try:
            accent = LABEL_MAP['accent'][self.dataset[item]['accent']]
            domain = LABEL_MAP['domain'][self.dataset[item]['domain']]
        except:
            accent=79
            domain=2
        vad = self.dataset[item].get('vad', 'speech')

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

        return (audio, text, audio_path, accent, domain, vad)

    
    

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
        domain = "general"
        vad = "speech"
        
        audio = np.asarray(audio)
        if 'whisper' in self.model_id and os.path.isdir(self.model_id):
            input_features = processor(
                audio,
                sampling_rate=AudioConfig.sr, 
                return_tensors="pt",
            )
            audio = input_features.input_features.squeeze()
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

        return (audio, text, audio_path, accent, domain, vad)

    
def transcribe_whisper(args, model, loader, split):
    tsince = int(round(time.time()))
    hypotheses = []
    references = []
    paths = []
    task_tags = []
    accents = []
    accent_preds = []
    domain_preds = []
    accent_ref = []
    domain_ref = []
    
    if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
    options = whisper.DecodingOptions(language="en", fp16=args.gpu > -1,
                                      without_timestamps=True)

    # options = dict(language=language, beam_size=5, best_of=5)
    # transcribe_options = dict(task="transcribe", **options)
    # transcription = model.transcribe(audio, **transcribe_options)["text"]

    for audio_or_mels, texts, audio_path, accent, domain, vad in tqdm(loader):
        if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
            audio_or_mels = audio_or_mels.to(device)
            with torch.no_grad():
                pred_ids = model.generate(audio_or_mels)
            results = processor.batch_decode(pred_ids, skip_special_tokens = True)
        elif 'whisper' in args.model_id_or_path:
            results = model.decode(audio_or_mels, options)
            results = [result.text for result in results]
        else:
            audio_or_mels = audio_or_mels.to(device)
            with torch.no_grad():
                logits = model(audio_or_mels).logits
                
            #asr
            asr_pred_ids = torch.argmax(torch.tensor(logits[0]), dim=-1)
            results = processor.batch_decode(asr_pred_ids)
            
            accent_pred_ids = torch.argmax(torch.tensor(logits[1]), dim=-1)
            domain_pred_ids = torch.argmax(torch.tensor(logits[2]), dim=-1)

        
        if "<|" in results[0]:
            task_tags.extend([get_task_tags(text) for text in results])
            hypotheses.extend([strip_task_tags(text) for text in results])
        else:
            hypotheses.extend(results)
        references.extend(texts)
        paths.extend(audio_path)
        accents.extend(accent)
        accent_preds.extend(accent_pred_ids.tolist())
        domain_preds.extend(domain_pred_ids.tolist())
        accent_ref.extend(accent.tolist())
        domain_ref.extend(domain.tolist())
        

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references,
                             audio_paths=paths, accent=accents, accent_preds=accent_preds, domain_preds=domain_preds, accent_ref=accent_ref, domain_ref=domain_ref))
    data["pred_clean"] = [clean_text(text) for text in data["hypothesis"]]
    data["ref_clean"] = [clean_text(text) for text in data["reference"]]
    data["pred_task_tags"] = task_tags if len(task_tags) > 0 else [''] * len(data)

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    macro_accent_f1 = metric.compute(
                predictions=data['accent_preds'], references=data['accent_ref'], average="macro"
            )
    micro_accent_f1 = metric.compute(
                predictions=data['accent_preds'], references=data['accent_ref'], average="micro"
            )
    
    macro_domain_f1 = metric.compute(
                predictions=data['domain_preds'], references=data['domain_ref'], average="macro"
            )
    micro_domain_f1 = metric.compute(
                predictions=data['domain_preds'], references=data['domain_ref'], average="micro"
            )
                
    print(f"Cleanup WER: {all_wer * 100:.2f} %")
    print(f"Macro accent head: {macro_accent_f1}")
    print(f"Micro accent head: {micro_accent_f1}")
    print(f"Macro domain head: {macro_domain_f1}")
    print(f"Micro domain head: {micro_domain_f1}")

    results = {
    "Cleanup_WER": f"{all_wer * 100:.2f}%",
    "Macro_accent_head": macro_accent_f1,
    "Micro_accent_head": micro_accent_f1,
    "Macro_domain_head": macro_domain_f1,
    "Micro_domain_head": micro_domain_f1
    }
    #import pdb; pdb.set_trace()
    with open(os.path.join(f"./results/", f"{args.data_csv_path.split('/')[1].split('.')[0]}_infe_results.json"), 'w') as f:
        json.dump(results, f)


    normalizer = EnglishTextNormalizer()

    gt_normalized, pred_normalized = [], []
    for i, (gt_text, pred_text) in enumerate(zip(data["reference"], data["hypothesis"])):
        gt = normalizer(gt_text)
        pred = normalizer(pred_text)
        if gt != "":
            gt_normalized.append(gt)
            pred_normalized.append(pred)

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
        processor = WhisperProcessor.from_pretrained(args.model_id_or_path)
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id_or_path)
    elif "whisper" in args.model_id_or_path:
        whisper_model = args.model_id_or_path.split("_")[1]
        model = whisper.load_model(whisper_model)
        print(
            f"Model {whisper_model} is {'multilingual' if model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )
#     elif DISCRIMINATIVE in args.model_id_or_path:
#         processor = Wav2Vec2Processor.from_pretrained(args.model_id_or_path)
#         model = Wav2Vec2ForCTCnCLS.from_pretrained(args.model_id_or_path).to(device)
    else:
        processor = Wav2Vec2Processor.from_pretrained(args.model_id_or_path)
        model = Wav2Vec2ForCTCnCLS.from_pretrained(args.model_id_or_path).to(device)

    model = model.to(device)
    model.eval()

    transcribe_whisper(args, model, data_loader, split)