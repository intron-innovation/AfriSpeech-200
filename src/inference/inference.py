import os

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/'
os.environ['XDG_CACHE_HOME'] = '/data/.cache/'
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import time
from datasets import load_metric, Dataset
import librosa
import torch
from transformers import AutoProcessor, AutoModelForCTC
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    Wav2Vec2Processor,
    set_seed,
)
import whisper
from src.utils.utils import write_pred, write_pred_inference_df
from src.utils.text_processing import clean_text
from src.utils.audio_processing import load_audio_file

set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wer_metric = load_metric("wer")

MODEL = None
PROCESSOR = None
SAMPLING_RATE = 16000

# This is the max audio length for whisper
MAX_MODEL_AUDIO_LEN_SECS = 87


def compute_benchmarks(batch):
    """
    Run inference using model on batch
    :param batch:
    :return:
    """
    speech = load_audio_file(batch["audio_paths"])

    input_features = PROCESSOR(
        speech, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt"
    )
    input_val = input_features.input_values.to(device)
    with torch.no_grad():
        logits = MODEL(input_val).logits
        batch["logits"] = logits

    pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
    pred = PROCESSOR.batch_decode(pred_ids)[0]
    batch["predictions"] = pred
    batch["predictions"] = clean_text(pred)
    batch["reference"] = clean_text(batch["text"]).lower()
    batch["wer"] = wer_metric.compute(
        predictions=[batch["predictions"]], references=[batch["reference"]]
    )
    return batch


def transcribe_whisper(dataframe, model_size="medium"):
    """
    Transcribe using whisper model
    :param dataframe: pd.DataFrame
    :param model_size: str
    :return: DataFrame
    """
    audio_paths = dataframe["audio_paths"].to_numpy()
    indexes = dataframe["idx"].to_numpy()
    texts = dataframe["text"].to_numpy()

    # Reset dataframe index
    dataframe.set_index("idx", inplace=True)

    model = whisper.load_model(model_size)
    for i in range(len(dataframe)):
        print(f"audio path: {i} -{audio_paths[i]}-")
        pred = model.transcribe(audio_paths[i], language="en")["text"]

        idx = indexes[i]
        dataframe.loc[idx, "reference"] = clean_text(texts[i]).lower()
        dataframe.loc[idx, "predictions"] = clean_text(pred)
        dataframe.loc[idx, "predictions_raw"] = pred.lower()
        dataframe.loc[idx, "wer_raw"] = wer_metric.compute(
            predictions=[pred.lower()],
            references=[texts[i].lower()],
        )
        dataframe.loc[idx, "wer"] = wer_metric.compute(
            predictions=[clean_text(pred)],
            references=[clean_text(texts[i]).lower()],
        )
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def run_benchmarks(model_id_or_path, test_dataset, output_dir="./results", gpu=-1):
    """
    Pipeline for running benchmarks for huggingface models on dev/test data
    :param output_dir: str
    :param model_id_or_path: str
    :param test_dataset: Dataset instance or pd.DataFrame
    :return:
    """
    global MODEL, PROCESSOR, device
    tsince = int(round(time.time()))
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu>-1) else "cpu")
    output_cols = None

    if "whisper" in model_id_or_path:
        test_dataset = transcribe_whisper(
            dataframe=test_dataset, model_size=model_id_or_path.split("_")[1]
        )
        test_dataset = Dataset.from_pandas(test_dataset)
        output_cols = [
            "audio_paths",
            "text",
            "predictions_raw",
            "reference",
            "predictions",
            "wer_raw",
            "wer",
            "accent",
        ]

    else:
        PROCESSOR = AutoProcessor.from_pretrained(model_id_or_path)
        MODEL = AutoModelForCTC.from_pretrained(model_id_or_path).to(device)
        test_dataset = test_dataset.map(compute_benchmarks)

    # test_dataset = compute_wer_dataset(test_dataset)
    n_samples = len(test_dataset)
    all_wer = wer_metric.compute(
        predictions=test_dataset["predictions"],
        references=test_dataset["reference"],
    )
    print(f"all_wer: {all_wer:0.03f}")
    write_pred(
        model_id_or_path, test_dataset, all_wer, cols=output_cols, output_dir=output_dir
    )
    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / n_samples:.4f}s per sample"
    )


def compute_wer_df(test_dataset, cols=None):
    if cols is None:
        cols = ["audio_paths", "text", "reference", "predictions", "wer", "accent"]
    predictions_data = {col: test_dataset[col] for col in cols}
    predictions_df = pd.DataFrame(data=predictions_data)
    predictions_df["predictions"] = [clean_text(pred) for pred in predictions_df["predictions"]]
    predictions_df["reference"] = [clean_text(ref) for ref in predictions_df["text"]]
    wers = []
    for i, row in predictions_df.iterrows():
        wer_metric.compute(
            predictions=[row["predictions"]], references=[row["reference"]]
        )
    predictions_df["wer"] = wers
    return predictions_df


def compute_wer_dataset(test_dataset):
    test_dataset["predictions"] = [clean_text(pred) for pred in test_dataset["predictions"]]
    test_dataset["reference"] = [clean_text(ref) for ref in test_dataset["text"]]
    wers = []
    for i in range(len(test_dataset)):
        wer_metric.compute(
            predictions=[test_dataset["predictions"][i]], references=[test_dataset["reference"][i]]
        )
    test_dataset["wer"] = wers
    return test_dataset
