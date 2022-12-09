import os
import argparse
import pandas as pd
from pathlib import Path
from functools import partial
from datasets import load_metric, Dataset, Audio
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from src.utils.utils import cleanup
from src.inference.inference import write_pred

set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_argument():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--audio_dir", type=str, default="./data/", help="directory to locate the audio")
    parser.add_argument("--model_id_or_path", type=str, default="openai/whisper-medium", 
                        help="id of the model or path to huggyface model")
    parser.add_argument("--output_dir", type=str, default="./whisper-medium-finetune", 
                        help="directory to store results")
    parser.add_argument("--train", action='store_true', help="Whether or not to train the model")
    parser.add_argument("--evaluate", action='store_true', help="Whether or not to evaluate the trained the model")
    parser.add_argument("--max_audio_len_secs", type=int, default=17,
                        help="maximum audio length passed to the inference model should")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="sampling rate of audio")
    # fmt: on

    args = parser.parse_args()
    return args


def load_data(
    data_path,
    audio_dir="./data/",
    duration=None,
    accent=None,
    age_group=None,
    country=None,
    origin=None,
    domain=None,
):
    """
    load train/dev/test data from csv path.
    :param max_audio_len_secs: int
    :param data_path: str
    :return: Dataset instance
    """
    data = pd.read_csv(audio_dir + data_path)
    data["audio_paths"] = data["audio_paths"].apply(
        lambda x: x.replace("/AfriSpeech-100/", audio_dir)
    )

    if duration is not None:
        data = data[data.duration < duration]

    if accent is not None:
        data = data[data.accent == accent]

    if country is not None:
        data = data[data.age_group == age_group]

    if origin is not None:
        data = data[data.origin == origin]

    if domain is not None:
        data = data[data.domain == domain]

    data["text"] = data["transcript"]
    print("before dedup", data.shape)
    data.drop_duplicates(subset=["audio_paths"], inplace=True)
    print("after dedup", data.shape)

    return Dataset.from_pandas(data)


def transcribe_whisper(batch, processor, model, metric, sampling_rate, device):
    """
    Run inference on batch using whisper model
    :param batch:
    :return:
    """

    speech, sampling_rate = librosa.load(batch["audio_paths"], sr=sampling_rate)
    input_features = processor(
        speech, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(device)
    generated_ids = model.generate(inputs=input_features)
    predicted_transcript = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    batch["predictions"] = predicted_transcript
    batch["clean_predictions"] = cleanup(predicted_transcript)
    batch["clean_text"] = cleanup(batch["text"])
    batch["clean_wer"] = metric.compute(
        predictions=[batch["clean_predictions"]], references=[batch["clean_text"]]
    )
    batch["wer"] = metric.compute(
        predictions=[batch["predictions"]], references=[batch["text"]]
    )
    return batch


def prepare_dataset(batch, feature_extractor, tokenizer):
    # Load and resample audio data to 16KHz
    audio = batch["audio_paths"]

    # Compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding wit -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it is appended later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # We do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    """Run main script"""
    args = parse_argument()

    if args.train:
        # Make output directory if does not already exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Load the dataset
        # fmt: off
        train_dataset = load_data(
            data_path="intron-train-public-58001.csv",
            audio_dir=args.audio_dir, 
            duration=args.max_audio_len_secs
        )       
        dev_dataset = load_data(
            data_path="intron-dev-public-3232.csv",
            audio_dir=args.audio_dir,
            duration=args.max_audio_len_secs
        )

        # Process the audio
        train_dataset = train_dataset.cast_column("audio_paths", Audio(sampling_rate=args.sampling_rate))
        dev_dataset = dev_dataset.cast_column("audio_paths", Audio(sampling_rate=args.sampling_rate))

        # Define processor, feature extractor, tokenizer and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = WhisperProcessor.from_pretrained(args.model_id_or_path, language="en", task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id_or_path)
        tokenizer = WhisperTokenizer.from_pretrained(args.model_id_or_path, language="en", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id_or_path)
        
        # Prepare dataset for training
        prepare_dataset = partial(prepare_dataset, feature_extractor=feature_extractor, tokenizer=tokenizer)
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
        dev_dataset = dev_dataset.map(prepare_dataset, remove_columns=dev_dataset.column_names)
        print(train_dataset, dev_dataset)
        # fmt: on

        # Override generation arguments
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        # Instantiate data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        # Load metric
        metric = load_metric("wer")
        compute_metrics = partial(compute_metrics, metric=metric)

        # Define the training configuration
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=5,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=500,
            num_train_epochs=5,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=25,
            save_steps=100,
            eval_steps=100,
            logging_steps=25,
            report_to="tensorboard",
            load_best_model_at_end=True,
            greater_is_better=False,
            push_to_hub=False,
        )

        # # Define the trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        # Save the processor object once before starting to train
        processor.save_pretrained(training_args.output_dir)
        args.model_id_or_path = Path(args.output_dir+"/checkpoint-5")

        # Train the model
        trainer.train()

    if args.evaluate:
        args.model_id_or_path = Path(args.model_id_or_path)

        # Make output directory if does not already exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Define processor, feature extractor, tokenizer and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dev dataset
        dev_dataset = load_data(
            data_path="intron-dev-public-3232.csv",
            audio_dir=args.audio_dir,
            duration=args.max_audio_len_secs,
        )

        # Define processor and model
        processor = WhisperProcessor.from_pretrained(args.model_id_or_path.parents[0], language="en", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id_or_path)

        # Define metric
        metric = load_metric("wer")

        # Run inference
        transcribe_whisper = partial(
            transcribe_whisper,
            processor=processor,
            model=model,
            metric=metric,
            sampling_rate=args.sampling_rate,
            device=device,
        )
        dev_dataset = dev_dataset.map(transcribe_whisper)

        all_wer = metric.compute(
            predictions=dev_dataset["predictions"],
            references=dev_dataset["text"],
        )

        print(f"all_wer: {all_wer:0.03f}")

        # fmt: off
        output_cols = [
            "audio_paths", "accent", "text", "predictions", 
            "clean_wer", "clean_text", "clean_predictions", "wer"
        ]
        # fmt: on

        # Write prediction to output folder
        write_pred(
            str(args.model_id_or_path),
            dev_dataset,
            all_wer,
            cols=output_cols,
            output_dir=args.output_dir,
        )
