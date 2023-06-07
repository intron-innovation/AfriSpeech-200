###### Code adapted from  ######
# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2
# https://github.com/openai/whisper
################################

import os

data_home = "data3"
os.environ["TRANSFORMERS_CACHE"] = f"/{data_home}/.cache/"
os.environ["XDG_CACHE_HOME"] = f"/{data_home}/.cache/"

import numpy as np
import torch
import time
import pandas as pd
import whisper
import json, codecs
import jiwer
from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoModelForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_afri_speech_data, DISCRIMINATIVE
from src.utils.text_processing import clean_text, strip_task_tags, get_task_tags
from src.utils.utils import parse_argument, write_pred_inference_df
from src.train.models import Wav2Vec2ForCTCnCLS

processor = None
device = None


class AfriSpeechWhisperDataset(torch.utils.data.Dataset):
    """
    A simple class to wrap AfriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(
        self,
        data_path,
        split="dev",
        device="cpu",
        model_id="whisper",
        max_audio_len_secs=17,
        audio_dir=f"./{data_home}/",
        gpu=-1,
    ):
        self.dataset = load_afri_speech_data(
            data_path=data_path,
            max_audio_len_secs=max_audio_len_secs,
            audio_dir=audio_dir,
            split=split,
            gpu=gpu,
        )
        self.device = device
        self.model_id = model_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio_path = self.dataset[item]["audio_paths"]
        text = self.dataset[item]["text"]
        accent = self.dataset[item]["accent"] if self.dataset[item]["accent"] is not None else "Unknown"
        domain = self.dataset[item]["domain"]
        vad = self.dataset[item].get("vad", "speech")
        country = self.dataset[item]["country"] if self.dataset[item]["country"] is not None else "Unknown"
        audio_ids = self.dataset[item]["audio_ids"]



        audio = load_audio_file(audio_path)
        if "whisper" in self.model_id and os.path.isdir(args.model_id_or_path):
            input_features = processor(
                audio, sampling_rate=AudioConfig.sr, return_tensors="pt",
            )
            audio = input_features.input_features.squeeze()
        elif "whisper" in self.model_id:
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = processor(
                audio,
                sampling_rate=AudioConfig.sr,
                padding="max_length",
                max_length=AudioConfig.sr * 17,
                truncation=True,
            )
            audio = input_features.input_values[0]

        return (audio, text, audio_path, accent, domain, vad, country, audio_ids)


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        split="test",
        device="cpu",
        model_id="whisper",
        max_audio_len_secs=17,
        gpu=-1,
    ):
        self.dataset = load_dataset("librispeech_asr", "clean", split=split)
        self.device = device
        self.model_id = model_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio = self.dataset[item]["audio"]["array"]
        text = self.dataset[item]["text"]
        accent = "US English"
        audio_path = self.dataset[item]["file"]
        domain = "general"
        vad = "speech"

        audio = np.asarray(audio)
        if "whisper" in self.model_id and os.path.isdir(self.model_id):
            input_features = processor(
                audio, sampling_rate=AudioConfig.sr, return_tensors="pt",
            )
            audio = input_features.input_features.squeeze()
        elif "whisper" in self.model_id:
            audio = np.asarray(audio, dtype=np.float32)
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = processor(
                audio,
                sampling_rate=AudioConfig.sr,
                padding="max_length",
                max_length=AudioConfig.sr * 17,
                truncation=True,
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
    if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )
    options = whisper.DecodingOptions(
        language="en", fp16=args.gpu > -1, without_timestamps=True
    )

    
    with codecs.open(
        f"src/experiments/{split}_accent_embeddings_7_2_1.json", "w", encoding="utf-8"
    ) as jsonf:
        for audio_or_mels, texts, audio_path, accent, domain, vad, country, audio_ids in tqdm(loader):
            if "whisper" in args.model_id_or_path and os.path.isdir(
                args.model_id_or_path
            ):
                audio_or_mels = audio_or_mels.to(device)
                with torch.no_grad():
                    pred_ids = model.generate(audio_or_mels)
                results = processor.batch_decode(pred_ids, skip_special_tokens=True)
            elif "whisper" in args.model_id_or_path:
                results = model.decode(audio_or_mels, options)
                results = [result.text for result in results]
            else:
                audio_or_mels = audio_or_mels.to(device)
                with torch.no_grad():
                    tick = model(audio_or_mels)
                    logits = model(audio_or_mels).logits[0]
                    accent_embedding = (
                        model(audio_or_mels).logits[1].detach().cpu().numpy()
                    )
                    for i in range(len(accent_embedding)):
                        accent_embedding_data = {
                           "audio_ids": audio_ids[i],
                            "accent": accent[i],
                            "embeddings": accent_embedding[i].tolist(),
                            "domain": domain[i],
                            "country": country[i],
                        }
                        jsonf.write(json.dumps(accent_embedding_data,) + "\n")
            #break




if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu"
    )

    if "librispeech" in args.data_csv_path:
        dataset = LibriSpeechDataset(
            data_path="librispeech_asr",
            split="test",
            model_id=args.model_id_or_path,
            device=device,
        )
        split = "test-libri-speech"

    else:
        split = "train"#args.data_csv_path.split("-")[1]
        dataset = AfriSpeechWhisperDataset(
            data_path=args.data_csv_path,
            max_audio_len_secs=args.max_audio_len_secs,
            audio_dir=args.audio_dir,
            device=device,
            split=split,
            gpu=args.gpu,
            model_id=args.model_id_or_path,
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


#   python3 -m src.inference.run_accent_embedding --model_id_or_path /data/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-multi-task-3-heads-weighted-7-2-1/checkpoints/ --gpu 1 --batchsize 8 --audio_dir /data/data/intron/ --data_csv data/intron-dev-public-3231-clean.csv
