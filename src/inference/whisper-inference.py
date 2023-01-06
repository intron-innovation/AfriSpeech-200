###### Code adapted from  ######
# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2
# https://github.com/openai/whisper
################################

import os
import numpy as np
import torch
import time
import pandas as pd
import whisper
import jiwer
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC

from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_afri_speech_data
from src.utils.text_processing import clean_text
from src.utils.utils import parse_argument

processor = None


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
        if 'whisper' in self.model_id:
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = processor(
                audio, sampling_rate=AudioConfig.sr, padding='max_length', return_tensors="pt"
            )
            audio = input_features.input_values.to(self.device)

        return (audio, text, audio_path, accent)


def transcribe_whisper(args, model, loader, model_id=None):
    tsince = int(round(time.time()))
    hypotheses = []
    references = []
    paths = []
    accents = []
    options = whisper.DecodingOptions(language="en", fp16=args.gpu > -1,
                                      without_timestamps=True)

    # options = dict(language=language, beam_size=5, best_of=5)
    # transcribe_options = dict(task="transcribe", **options)
    # transcription = model.transcribe(audio, **transcribe_options)["text"]

    for audio_or_mels, texts, audio_path, accent in tqdm(loader):
        if 'whisper' in model_id:
            results = model.decode(audio_or_mels, options)
        else:
            with torch.no_grad():
                logits = model(audio_or_mels).logits
            pred_ids = torch.argmax(torch.tensor(logits), dim=-1)
            results = processor.batch_decode(pred_ids)[0]

        hypotheses.extend([result.text for result in results])
        references.extend(texts)
        paths.extend(audio_path)
        accents.extend(accent)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references,
                             audio_paths=paths, accent=accents))

    data["pred_clean"] = [clean_text(text) for text in data["hypothesis"]]
    data["ref_clean"] = [clean_text(text) for text in data["reference"]]

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    print(f"Cleanup WER: {all_wer * 100:.2f} %")

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

    n_samples = len(loader.dataset)
    split = args.data_csv_path.split("-")[1]
    output_path = f"{args.output_dir}/intron-open-{split}-{args.model_id_or_path}-wer-{round(all_wer, 4)}-{n_samples}.csv"
    data.to_csv(output_path, index=False)
    print(output_path)

    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{args.model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / n_samples:.4f}s per sample"
    )


if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu")

    dataset = AfriSpeechWhisperDataset(data_path=args.data_csv_path,
                                       max_audio_len_secs=args.max_audio_len_secs,
                                       audio_dir=args.audio_dir, device=device,
                                       split=args.data_csv_path.split("-")[1],
                                       gpu=args.gpu, model_id=args.model_id_or_path
                                       )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize)

    if "whisper" in args.model_id_or_path:
        whisper_model = args.model_id_or_path.split("_")[1]  # "base.en"
        model = whisper.load_model(whisper_model)
        print(
            f"Model {whisper_model} is {'multilingual' if model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model_id_or_path)
        model = AutoModelForCTC.from_pretrained(args.model_id_or_path).to(device)

    model.to(device)

    transcribe_whisper(args, model, data_loader, args.model_id_or_path)
