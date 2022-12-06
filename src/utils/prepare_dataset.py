import logging
import os
import time
import json
import sys
from datetime import datetime
import pandas as pd
import subprocess

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/'

from datasets import load_dataset, load_metric
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
import librosa
import torch
from transformers import (
    Wav2Vec2Tokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from src.utils.audio_processing import AudioConfig, load_audio_file
from src.utils.text_processing import clean_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging_level = logging.DEBUG
logger.setLevel(logging_level)

PROCESSOR = None
CONFIG = None


class DataConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def data_prep(config):
    # Prepare data for the model
    global CONFIG, PROCESSOR
    CONFIG = config
    start = time.time()

    raw_dataset = load_data(config.train_path, config.val_path)
    logger.debug(f"...Data Read Complete in {time.time() - start:.4f}. Starting Tokenizer...")

    vocab_file_name = load_vocab(config.model_path, config.ckpt_path, config.exp_dir, raw_dataset)
    PROCESSOR = load_processor(vocab_file_name)
    logger.debug(f"...Load vocab and processor complete in {time.time() - start:.4f}.\n"
                 f"Loading dataset...")

    train_dataset = CustomASRDataset(config.train_path, transform_audio, transform_labels, 
                                     config.audio_path, 'train', config.max_audio_len_secs)
    val_dataset = CustomASRDataset(config.val_path, transform_audio, transform_labels, 
                                   config.audio_path, 'dev', config.max_audio_len_secs)
    logger.debug(f"Load train and val dataset done in {time.time() - start:.4f}.")
    return train_dataset, val_dataset, PROCESSOR


def load_vocab(model_path, checkpoints_path, exp_dir, raw_datasets):
    create_new_vocab = False
    vocab_file_name = None

    if os.path.isdir(model_path) and 'vocab.json' in os.listdir(model_path):
        vocab_files = ['preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json']
        for v in vocab_files:
            subprocess.call(['cp', os.path.join(model_path, v), os.path.join(checkpoints_path, v)])
        vocab_file_name = os.path.join(checkpoints_path, 'vocab.json')
        if os.path.isfile(vocab_file_name):
            print(f"vocab detected at {vocab_file_name}")
        else:
            create_new_vocab = True

    elif os.path.isdir(checkpoints_path) and len(os.listdir(checkpoints_path)) > 0:
        vocab_file_name = [x for x in os.listdir(checkpoints_path) if 'vocab' in x]
        if len(vocab_file_name) > 0:
            vocab_file_name = os.path.join(checkpoints_path, vocab_file_name[0])
            print(f"vocab detected at {vocab_file_name}")
        else:
            create_new_vocab = True
    else:
        create_new_vocab = True

    if create_new_vocab:
        vocab_dict = prepare_tokenizer(raw_datasets)
        vocab_file_name = f'vocab-{datetime.now().strftime("%d-%m-%Y--%H:%M:%S")}.json'
        vocab_file_name = os.path.join(exp_dir, 'checkpoints', vocab_file_name)
        logger.debug(f"creating new vocab {vocab_file_name}")
        with open(vocab_file_name, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
    elif vocab_file_name:
        with open(vocab_file_name, 'r') as vocab_file:
            vocab_dict = json.load(vocab_file)
    else:
        vocab_dict = {}

    logger.info(f"---vocab dict: {len(vocab_dict)}\n{vocab_dict}")
    return vocab_file_name


def load_data(train_path, val_path):
    return load_dataset('csv', data_files={'train': train_path, 'val': val_path})


def remove_special_characters(batch):
    batch['transcript'] = clean_text(batch['transcript']) + " "
    return batch


def extract_chars_vocab(batch):
    all_text = " ".join(batch['transcript'])
    vocab = list(set(all_text))
    return {'vocab': [vocab], 'all_text': [all_text]}


def special_tokens(vocab_dict):
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


def prepare_tokenizer(raw_datasets):
    raw_datasets = raw_datasets.map(remove_special_characters, num_proc=6)
    vocabs = raw_datasets.map(extract_chars_vocab,
                              batched=True, batch_size=-1, keep_in_memory=True,
                              remove_columns=raw_datasets.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["val"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict = special_tokens(vocab_dict)
    return vocab_dict


def get_feature_extractor():
    return Wav2Vec2FeatureExtractor(feature_size=1,
                                    sampling_rate=AudioConfig.sr,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=True)


def load_processor(vocab_file_name):
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file_name, unk_token="[UNK]",
                                     pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = get_feature_extractor()
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def transform_audio(audio_path):
    try:
        speech = load_audio_file(audio_path)
    except Exception as e:
        print(e)
        speech, fs = librosa.load('/data/data/intron/e809b58c-4f05-4754-b98c-fbf236a88fbc/544bbfe5e1c6f8afb80c4840b681908d.wav',
                                  sr=AudioConfig.sr)

    return PROCESSOR(speech, sampling_rate=AudioConfig.sr).input_values


def transform_labels(text):
    text = clean_text(text)
    with PROCESSOR.as_target_processor():
        labels = PROCESSOR(text.lower()).input_ids
    return labels


class CustomASRDataset(Dataset):
    def __init__(self, data_file, transform=None, transform_target=None, audio_dir=None, 
                 split='train', max_audio_len_secs=-1):
        self.asr_data = pd.read_csv(data_file)
        if max_audio_len_secs != -1:
            self.asr_data = self.asr_data[self.asr_data.duration < max_audio_len_secs]
        self.asr_data = self.asr_data[self.asr_data.transcript.str.len() >= 10]
        self.asr_data["audio_paths"] = self.asr_data["audio_paths"].apply(
            lambda x: x.replace(f"/AfriSpeech-100/{split}/", audio_dir)
        )
        self.transform = transform
        self.target_transform = transform_target

    def __len__(self):
        return len(self.asr_data)

    def __getitem__(self, idx):
        audio_path = self.asr_data.iloc[idx, 8]  # audio_path
        text = self.asr_data.iloc[idx, 5]  # transcript
        input_audio = self.transform(audio_path)
        label = self.target_transform(text)

        return {'input_values': input_audio[0], 'labels': label, 'input_lengths': len(input_audio[0])}


@dataclass
class DataCollatorCTCWithPaddingGroupLen:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch
