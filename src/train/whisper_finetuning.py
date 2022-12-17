#
# Code mostly borrowed from https://huggingface.co/blog/fine-tune-whisper 🤗
#

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
    WhisperConfig
)
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.train.train import parse_argument, train_setup, get_checkpoint, data_setup
from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_custom_dataset
from src.utils.text_processing import clean_text
from src.utils.utils import cleanup
from src.inference.inference import write_pred

set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(
    data_path,
    audio_dir="./data/",
    split="train",
    duration=None,
    accent=None,
    age_group=None,
    country=None,
    origin=None,
    domain='all',
    min_transcript_len=None
):
    """
    load train/dev/test data from csv path.
    :param split: str
    :param min_transcript_len:
    :param domain: str
    :param origin: str
    :param country: str
    :param age_group: str
    :param accent: str
    :param duration: str
    :param audio_dir: str
    :param data_path: str
    :return: Dataset instance
    """
    data = pd.read_csv(data_path)
    data["audio_paths"] = data["audio_paths"].apply(
        lambda x: x.replace(f"/AfriSpeech-100/{split}/", audio_dir)
    )

    if duration:
        data = data[data.duration < duration]

    if min_transcript_len:
        data = data[data.transcript.str.len() >= min_transcript_len]

    if accent:
        data = data[data.accent == accent]

    if country:
        data = data[data.age_group == age_group]

    if origin:
        data = data[data.origin == origin]

    if domain != 'all':
        data = data[data.domain == domain]

    data["text"] = data["transcript"]
    print("before dedup", data.shape)
    data.drop_duplicates(subset=["audio_paths"], inplace=True)
    print("after dedup", data.shape)

    return Dataset.from_pandas(data)


def transcribe_whisper(batch, processor, model, metric, sampling_rate, device):
    """
    run inference on batch using whisper model
    :param batch: Dataset instance
    :param processor: object
    :param model: object
    :param metric: object
    :param sampling_rate: int
    :param device: str
    :return: Dataset instance
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
    """
    compute metrics
    :param pred: Dataset instance
    :param metric: object
    :return: dict
    """
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
    args, config = parse_argument()
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)

    # Define processor, feature extractor, tokenizer and model
    processor = WhisperProcessor.from_pretrained(config['models']['model_path'], language="en", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config['models']['model_path'])
    tokenizer = WhisperTokenizer.from_pretrained(config['models']['model_path'], language="en", task="transcribe")

    def transform_whisper_audio(audio_path):
        try:
            speech = load_audio_file(audio_path)
        except Exception as e:
            print(e)
            speech, fs = librosa.load(
                '/data/data/intron/e809b58c-4f05-4754-b98c-fbf236a88fbc/544bbfe5e1c6f8afb80c4840b681908d.wav',
                sr=AudioConfig.sr)

        return feature_extractor(speech, sampling_rate=AudioConfig.sr).input_values


    def transform_whisper_labels(text):
        text = clean_text(text)
        return tokenizer(text.lower()).input_ids

    # Load the dataset
    # fmt: off
    # dev_dataset = load_data(
    #     data_path=config['data']['val'],
    #     audio_dir=config['audio']['audio_path'],
    #     split="dev",
    #     duration=float(config['hyperparameters']['max_audio_len_secs']),
    #     min_transcript_len=float(config['hyperparameters']['min_transcript_len']),
    #     domain=config['data']['domain']
    # )
    dev_dataset = load_custom_dataset(data_config, 'dev', transform_whisper_audio, transform_whisper_labels)
    train_dataset = load_custom_dataset(data_config, 'train', transform_whisper_audio, transform_whisper_labels)

    sampling_rate = int(config['hyperparameters']['sampling_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])
    print(f"model starting...from last checkpoint:{last_checkpoint}")

    if config['hyperparameters']['do_train'] == "True":
        # train_dataset = load_data(
        #     data_path=config['data']['train'],
        #     audio_dir=config['audio']['audio_path'],
        #     split="train",
        #     duration=float(config['hyperparameters']['max_audio_len_secs']),
        #     min_transcript_len=float(config['hyperparameters']['min_transcript_len']),
        #     domain=config['data']['domain']
        # )

        # Process the audio
        # train_dataset = train_dataset.cast_column("audio_paths", Audio(sampling_rate=sampling_rate))
        # dev_dataset = dev_dataset.cast_column("audio_paths", Audio(sampling_rate=sampling_rate))

        # Prepare dataset for training
        # prepare_dataset = partial(prepare_dataset, feature_extractor=feature_extractor, tokenizer=tokenizer)
        # train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
        # dev_dataset = dev_dataset.map(prepare_dataset, remove_columns=dev_dataset.column_names)
        print(train_dataset, dev_dataset)

        # load model
        w_config = WhisperConfig.from_pretrained(config['models']['model_path'], use_cache=False)
        model = WhisperForConditionalGeneration.from_pretrained(
            last_checkpoint if last_checkpoint else config['models']['model_path'],
            config=w_config
        )

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
            output_dir=checkpoints_path,
            overwrite_output_dir=True if config['hyperparameters']['overwrite_output_dir'] == "True" else False,
            group_by_length=True if config['hyperparameters']['group_by_length'] == "True" else False,
            length_column_name=config['hyperparameters']['length_column_name'],
            data_seed=int(config['hyperparameters']['data_seed']),
            per_device_train_batch_size=int(config['hyperparameters']['train_batch_size']),
            gradient_accumulation_steps=int(config['hyperparameters']['gradient_accumulation_steps']),
            learning_rate=float(config['hyperparameters']['learning_rate']),
            warmup_steps=int(config['hyperparameters']['warmup_steps']),
            num_train_epochs=int(config['hyperparameters']['num_epochs']),
            gradient_checkpointing=True if config['hyperparameters']['gradient_checkpointing'] == "True" else False,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="steps",
            per_device_eval_batch_size=int(config['hyperparameters']['val_batch_size']),
            predict_with_generate=True if config['hyperparameters']['predict_with_generate'] == "True" else False,
            generation_max_length=int(config['hyperparameters']['generation_max_length']),
            save_steps=int(config['hyperparameters']['save_steps']),
            eval_steps=int(config['hyperparameters']['eval_steps']),
            logging_steps=int(config['hyperparameters']['logging_steps']),
            report_to=config['hyperparameters']['report_to'],
            load_best_model_at_end=True if config['hyperparameters']['load_best_model_at_end'] == 'True' else False,
            metric_for_best_model='eval_wer',
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
        processor.save_pretrained(checkpoints_path)
        # args.model_id_or_path = Path(args.output_dir + "/checkpoint-5")

        # Train the model
        trainer.train(resume_from_checkpoint=checkpoint_)

        model.save_pretrained(checkpoints_path)
        processor.save_pretrained(checkpoints_path)

    if config['hyperparameters']['do_eval'] == "True":

        # Define processor, feature extractor, tokenizer and model
        # Define processor and model
        processor = WhisperProcessor.from_pretrained(
            config['models']['model_path'], language="en", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            last_checkpoint if last_checkpoint else config['models']['model_path']
        )

        # Define metric
        metric = load_metric("wer")

        # Run inference
        transcribe_whisper = partial(
            transcribe_whisper,
            processor=processor,
            model=model,
            metric=metric,
            sampling_rate=sampling_rate,
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
            config['models']['model_path'],
            dev_dataset,
            all_wer,
            cols=output_cols,
            output_dir=args.output_dir,
        )
