import pandas as pd
import time
from datasets import load_metric, Dataset
import librosa
import torch
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    Wav2Vec2Processor,
    set_seed,
)
import whisper
from src.utils.utils import cleanup

set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wer_metric = load_metric("wer")

MODEL = None
PROCESSOR = None
SAMPLING_RATE = 16000

# This is the max audio length for whisper
MAX_MODEL_AUDIO_LEN_SECS = 87


def load_data(
    data_path, max_audio_len_secs=17, audio_dir="./data/", return_dataset=True
):
    """
    load train/dev/test data from csv path.
    :param max_audio_len_secs: int
    :param data_path: str
    :return: Dataset instance
    """
    data = pd.read_csv(data_path)
    data["audio_paths"] = data["audio_paths"].apply(
        lambda x: x.replace("/AfriSpeech-100/dev/", audio_dir)
    )

    if max_audio_len_secs != -1:
        data = data[data.duration < max_audio_len_secs]
    else:
        # Check if any of the sample is longer than
        # the global MAX_MODEL_AUDIO_LEN_SECS
        if (data.duration.to_numpy() > MAX_MODEL_AUDIO_LEN_SECS).any():
            raise ValueError(
                f"Detected speech longer than {MAX_MODEL_AUDIO_LEN_SECS} secs"
                "-- set `max_audio_len_secs` to filter longer speech!"
            )

    data["text"] = data["transcript"]
    print("before dedup", data.shape)
    data.drop_duplicates(subset=["audio_paths"], inplace=True)
    print("after dedup", data.shape)
    if return_dataset:
        return Dataset.from_pandas(data)
    else:
        return data


def compute_benchmarks(batch):
    """
    Run inference using model on batch
    :param batch:
    :return:
    """
    speech, fs = librosa.load(batch["audio_paths"], sr=SAMPLING_RATE)
    if fs != SAMPLING_RATE:
        speech = librosa.resample(speech, fs, SAMPLING_RATE)

    input_features = PROCESSOR(
        speech, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt"
    )
    input_val = input_features.input_values.to(device)
    with torch.no_grad():
        logits = MODEL(input_val).logits
        batch["logits"] = logits

    pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
    pred = PROCESSOR.batch_decode(pred_ids)[0]
    batch["predictions"] = cleanup(pred)
    batch["reference"] = cleanup(batch["text"]).lower()
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
        pred = model.transcribe(audio_paths[i], language="en")["text"]

        idx = indexes[i]
        dataframe.loc[idx, "reference"] = cleanup(texts[i]).lower()
        dataframe.loc[idx, "predictions"] = cleanup(pred)
        dataframe.loc[idx, "predictions_raw"] = pred.lower()
        dataframe.loc[idx, "wer_raw"] = wer_metric.compute(
            predictions=[pred],
            references=[texts[i].lower()],
        )
        dataframe.loc[idx, "wer"] = wer_metric.compute(
            predictions=[cleanup(pred)],
            references=[cleanup(texts[i]).lower()],
        )
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def write_pred(model_id_or_path, results, wer, cols=None, output_dir="./results"):
    """
    Write model predictions to file
    :param cols: List[str]
    :param output_dir: str
    :param model_id_or_path: str
    :param results: Dataset instance
    :param wer: float
    :return: DataFrame
    """
    model_id_or_path = model_id_or_path.replace("/", "-")
    if cols is None:
        cols = ["audio_paths", "text", "reference", "predictions", "wer", "accent"]
    predictions_data = {col: results[col] for col in cols}
    predictions_df = pd.DataFrame(data=predictions_data)

    output_path = f"{output_dir}/intron-open-test-{model_id_or_path}-wer-{round(wer, 4)}-{len(predictions_df)}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(output_path)
    return predictions_df


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

    if "hubert" in model_id_or_path:
        PROCESSOR = Wav2Vec2Processor.from_pretrained(model_id_or_path)
        MODEL = HubertForCTC.from_pretrained(model_id_or_path).to(device)
        test_dataset = test_dataset.map(compute_benchmarks)

    elif "whisper" in model_id_or_path:
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
        PROCESSOR = Wav2Vec2Processor.from_pretrained(model_id_or_path)
        MODEL = Wav2Vec2ForCTC.from_pretrained(model_id_or_path).to(device)
        test_dataset = test_dataset.map(compute_benchmarks)

    n_samples = len(test_dataset)
    all_wer = wer_metric.compute(
        predictions=test_dataset["predictions"],
        references=test_dataset["reference"],
    )
    print(f"all_wer: {all_wer:0.03f}")
    write_pred(
        model_id_or_path, test_dataset, all_wer, cols=output_cols, output_dir=output_dir
    )
    ttime_elapsed = int(round(time.time())) - tsince
    print(
        f"{model_id_or_path}-- Inference Time: {ttime_elapsed / 60:.4f}m | "
        f"{ttime_elapsed / n_samples:.4f}s per sample"
    )