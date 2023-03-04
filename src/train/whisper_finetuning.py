#
# Code mostly borrowed from https://huggingface.co/blog/fine-tune-whisper 🤗

# Other sources
# https://huggingface.co/sanchit-gandhi/whisper-medium-switchboard-5k/blob/main/run_speech_recognition_whisper.py
# https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz
#

import os

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/'
os.environ['XDG_CACHE_HOME'] = '/data/.cache/'
os.environ["WANDB_DISABLED"] = "true"

import gc
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
    WhisperConfig,
    set_seed,
    Seq2SeqTrainingArguments,
)
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.train.train import parse_argument, train_setup, get_checkpoint, data_setup
from src.inference.inference import write_pred
from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_custom_dataset
from src.utils.text_processing import clean_text
from src.utils.sampler import IntronSeq2SeqTrainer

import logging
logging.disable(logging.WARN)

gc.collect()
torch.cuda.empty_cache()

set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_audio = '/data/data/intron/e809b58c-4f05-4754-b98c-fbf236a88fbc/544bbfe5e1c6f8afb80c4840b681908d.wav'
wer_metric = load_metric("wer")


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
        # print("labels before:", labels.shape, labels)
        # print("decode labels:", self.processor.tokenizer.batch_decode(labels))
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            # print("labels after:", labels)
            
        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    """
    compute metrics
    :param pred: Dataset instance
    :return: dict
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # We do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    preds = [x.lower() for x in pred_str]
    labels = [x.lower() for x in label_str]

    wer = wer_metric.compute(predictions=preds, references=labels)

    return {"wer": wer}


if __name__ == "__main__":
    """Run main script"""
    args, config = parse_argument()
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)

    # Define processor, feature extractor, tokenizer and model
    processor = WhisperProcessor.from_pretrained(config['models']['model_path'], language="en", task="transcribe")
    # language="english"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config['models']['model_path'])
    tokenizer = WhisperTokenizer.from_pretrained(config['models']['model_path'], language="en", task="transcribe")
    # language="english"

    def transform_dataset(audio_path, text):
        # Load and resample audio data to 16KHz
        try:           
            speech = load_audio_file(audio_path)
        except Exception as e:
            print(f"prepare: {audio_path} not found {str(e)}")
            speech = load_audio_file(temp_audio)
        
        # Compute log-Mel input features from input audio array
        audio = feature_extractor(speech, sampling_rate=AudioConfig.sr).input_features[0]

        # Encode target text to label ids
        text = clean_text(text)
        # print("text:", text)
        labels = tokenizer(text.lower()).input_ids
        # print("labels:", labels)

        return audio, labels

    # Load the dataset
    dev_dataset = load_custom_dataset(data_config, data_config.val_path, 
                                      'dev', transform_dataset, prepare=True)
    train_dataset = load_custom_dataset(data_config, data_config.train_path, 
                                        'train', transform_dataset, prepare=True)

    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])
    print(f"model starting...from last checkpoint:{last_checkpoint}")

    # load model
    model = WhisperForConditionalGeneration.from_pretrained(
        last_checkpoint if last_checkpoint else config['models']['model_path'],
    ).to(device)
    
    logging.disable(logging.WARN)
    
    if config['hyperparameters']['do_train'] == "True":

        # Override generation arguments
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        # to use gradient checkpointing
        model.config.use_cache = True if config['hyperparameters']['use_cache'] == "True" else False

        # Instantiate data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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
            #report_to=config['hyperparameters']['report_to'],
            load_best_model_at_end=True if config['hyperparameters']['load_best_model_at_end'] == 'True' else False,
            metric_for_best_model='eval_wer',
            greater_is_better=False,
            push_to_hub=False,
            logging_first_step=True,
            ignore_data_skip=True if config['hyperparameters']['ignore_data_skip'] == 'True' else False,
            dataloader_num_workers=int(config['hyperparameters']['dataloader_num_workers']),
            ddp_find_unused_parameters=True if config['hyperparameters']['ddp_find_unused_parameters'] == "True" else False,
        )

        # # Define the trainer
        trainer = IntronSeq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            sampler=config['data']['sampler'] if 'sampler' in config['data'] else None
        )

        # Save the processor object once before starting to train
        processor.save_pretrained(checkpoints_path)

        # Train the model
        trainer.train(resume_from_checkpoint=checkpoint_)

        model.save_pretrained(checkpoints_path)
        processor.save_pretrained(checkpoints_path)
    
