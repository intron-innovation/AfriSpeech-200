import os
import argparse
import pandas as pd
import time
from datasets import load_dataset, load_metric, Dataset
import librosa
import torch
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    Wav2Vec2Tokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from src.utils import cleanup

set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wer_metric = load_metric("wer")

model = None
processor = None
sampling_rate = 16000


def load_data(data_path, max_audio_len_secs=17):
    """
    load train/dev/test data from csv path.
    :param max_audio_len_secs: int
    :param data_path: str
    :return: Dataset instance
    """
    data = pd.read_csv(data_path)
    data = data[data.duration < max_audio_len_secs]
    data["text"] = data["transcript"]
    print("before dedup", data.shape)
    data.drop_duplicates(subset=["audio_paths"], inplace=True)
    print("after dedup", data.shape)
    return Dataset.from_pandas(data)


def compute_benchmarks(batch):
    """
    Run inference using model on batch
    :param batch:
    :return:
    """
    speech, fs = librosa.load(batch["audio_paths"], sr=sampling_rate)
    if fs != sampling_rate:
        speech = librosa.resample(speech, fs, sampling_rate)

    input_features = processor(
        speech, sampling_rate=sampling_rate, padding=True, return_tensors="pt"
    )
    input_val = input_features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_val).logits
        batch["logits"] = logits

    pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
    pred = processor.batch_decode(pred_ids)[0]
    batch["predictions"] = cleanup(pred)
    batch["reference"] = cleanup(batch["transcript"]).lower()
    batch["wer"] = wer_metric.compute(
        predictions=[batch["predictions"]], references=[batch["reference"]]
    )
    return batch


def write_pred(model_id_or_path, results, wer, output_dir="./results"):
    """
    Write model predictions to file
    :param output_dir: str
    :param model_id_or_path: str
    :param results: Dataset instance
    :param wer: float
    :return: DataFrame
    """
    model_id_or_path = model_id_or_path.replace("/", "-")
    cols = ["audio_paths", "transcript", "reference", "predictions", "wer", "accent"]
    predictions_data = {col: results[col] for col in cols}
    predictions_df = pd.DataFrame(data=predictions_data)

    output_path = f"{output_dir}/intron-open-test-{model_id_or_path}-wer-{round(wer, 4)}-{len(predictions_df)}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(output_path)
    return predictions_df


def run_benchmarks(model_id_or_path, test_dataset, output_dir="./results"):
    """
    Pipeline for running benchmarks for huggingface models on dev/test data
    :param model_id_or_path: str
    :param test_dataset: Dataset
    :return:
    """
    global model, processor
    tsince = int(round(time.time()))
    n_samples = len(test_dataset)
    processor = Wav2Vec2Processor.from_pretrained(model_id_or_path)
    if "hubert" in model_id_or_path:
        model = HubertForCTC.from_pretrained(model_id_or_path).to(device)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path).to(device)
    test_dataset = test_dataset.map(compute_benchmarks)
    all_wer = wer_metric.compute(
        predictions=test_dataset["predictions"], references=test_dataset["reference"]
    )
    print(f"all_wer: {all_wer:0.03f}")
    write_pred(model_id_or_path, test_dataset, all_wer, output_dir=output_dir)
    ttime_elapsed = int(round(time.time())) - tsince
    print(
        f"{model_id_or_path}-- Inference Time: {ttime_elapsed / 60:.4f}m | "
        f"{ttime_elapsed / n_samples:.4f}s per sample"
    )


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="./data/intron-dev-public-3232.csv",
        help="path to data csv file",
    )
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="facebook/hubert-large-ls960-ft",
        help="id of the model or path to huggyface model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="directory to store results"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Run main script"""
    args = parse_argument()

    # Make output directory if does not already exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset using huggyface dataset class because the
    # `compute_benchmarks` function is mapped in the `run_benchmarks`
    # function. So, `test_dataset` must be a Dataset type.
    ds = load_dataset("csv", data_files={"dev": args.data_csv_path})["dev"]

    # This block can be removed once the data csv file removes the
    # `/AfriSpeech-100` prepended to the audio path.
    DATA_PAR_DIR = os.path.dirname(args.data_csv_path)

    def _fix_audio_path(batch):
        batch["audio_paths"] = batch["audio_paths"].replace(
            "/AfriSpeech-100", f"{DATA_PAR_DIR}"
        )
        return batch

    ds = ds.map(_fix_audio_path).select(range(10))

    run_benchmarks(
        model_id_or_path=args.model_id_or_path,
        test_dataset=ds,
        output_dir=args.output_dir,
    )
