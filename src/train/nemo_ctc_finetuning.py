import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict
from pathlib import Path

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
import re
import unicodedata

from tqdm.auto import tqdm
import json
import pandas as pd

from collections import defaultdict

import torch
import torch.nn as nn

VERSION = "cv-corpus-6.1-2020-12-11"
LANGUAGE = "en"

tokenizer_dir = os.path.join('tokenizers', LANGUAGE)
manifest_dir = os.path.join('manifests', LANGUAGE)


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest

cmd = "ffmpeg -i "
cmd1 = " -ac 1 -ar 16000 "

def read_train_intron(path):
  final = []
  df = pd.read_csv(path)
  for i in range(df.shape[0]):
    data = {}
    data["audio_filepath"] = df.iloc[i]["audio_paths"].replace("/AfriSpeech-100/", "/scratch/pbsjobs/axy327/")
    data["duration"] = df.iloc[i]["duration"]
    data["text"] = df.iloc[i]["transcript"]
    data["domain"] = df.iloc[i]["domain"]
    os.system(cmd + data["audio_filepath"] + cmd1 + "/scratch/pbsjobs/" + data["audio_filepath"].split("/")[-1])
    os.system("mv /scratch/pbsjobs/" + data["audio_filepath"].split("/")[-1] + " " + data["audio_filepath"])
    my_file = Path(data["audio_filepath"])
    if data["duration"] <= 17 and len(data["text"]) >= 10 and my_file.exists():
      final.append(data)
  return final

def read_intron(path):
  final = []
  df = pd.read_csv(path)
  for i in range(df.shape[0]):
    data = {}
    data["audio_filepath"] = df.iloc[i]["audio_paths"].replace("/AfriSpeech-100/", "/scratch/pbsjobs/axy327/")
    data["duration"] = df.iloc[i]["duration"]
    data["text"] = df.iloc[i]["transcript"]
    data["domain"] = df.iloc[i]["domain"]
    # os.system(cmd + data["audio_filepath"] + cmd1 + "/scratch/pbsjobs/" + data["audio_filepath"].split("/")[-1])
    # os.system("mv /scratch/pbsjobs/" + data["audio_filepath"].split("/")[-1] + " " + data["audio_filepath"])
    my_file = Path(data["audio_filepath"])
    if data["duration"] <= 17 and len(data["text"]) >= 10 and my_file.exists():
      final.append(data)
  return final


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath

train_intron_manifest = read_intron("intron-train-public-58001.csv")
# train_intron_manifest = read_intron("intron-dev-public-3232.csv")
dev_intron_manifest = read_intron("intron-dev-public-3232.csv")

intron_train_text = [data['text'] for data in train_intron_manifest]
intron_dev_text = [data['text'] for data in dev_intron_manifest]

def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

intron_train_charset = get_charset(train_intron_manifest)
intron_dev_charset = get_charset(dev_intron_manifest)

intron_train_set = set(intron_train_charset.keys())
intron_dev_set = set(intron_dev_charset.keys())

print(f"Number of tokens in intron train set : {len(intron_train_set)}")
print(f"Number of tokens in intron dev set : {len(intron_dev_set)}")


test_oov = intron_dev_set - intron_train_set
print(f"Number of OOV tokens in test set : {len(test_oov)}")
print()
print(test_oov)

perform_dakuten_normalization = True #@param ["True", "False"] {type:"raw"}
PERFORM_DAKUTEN_NORMALIZATION = bool(perform_dakuten_normalization)

def process_dakuten(text):
    normalized_text = unicodedata.normalize('NFD', text)
    normalized_text = normalized_text.replace("\u3099", "").replace("\u309A", "")
    return normalized_text

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\{\}\【\】\・\。\『\』\、\ー\〜]'  # remove special character tokens

def remove_special_characters(data):
    data["text"] = re.sub(chars_to_ignore_regex, '', data["text"]).lower().strip()
    return data

def remove_dakuten(data):
    # perform dakuten normalization (if it was requested)
    if PERFORM_DAKUTEN_NORMALIZATION:
        text = data['text']
        data['text'] = process_dakuten(text)
    return data

## Process dataset

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest

# List of pre-processing functions
PREPROCESSORS = [
    remove_special_characters,
    remove_dakuten,
]

# Load manifests
intron_train_data = read_intron("intron-train-public-58001.csv")
# intron_train_data = read_intron("intron-dev-public-3232.csv")
intron_dev_data = read_intron("intron-dev-public-3232.csv")

# Apply preprocessing
intron_train_data_processed = apply_preprocessors(intron_train_data, PREPROCESSORS)
intron_dev_data_processed = apply_preprocessors(intron_dev_data, PREPROCESSORS)

# Write new manifests
intron_train_manifest_cleaned = write_processed_manifest(intron_train_data_processed, "intron-train-public-58001.json")
# intron_train_manifest_cleaned = write_processed_manifest(intron_dev_data_processed, "intron-dev-public-3232.json")
intron_dev_manifest_cleaned = write_processed_manifest(intron_dev_data_processed, "intron-dev-public-3232.json")

intron_train_data = read_intron("intron-train-public-58001.csv")
# intron_train_data = read_intron("intron-dev-public-3232.csv")
intron_dev_data = read_intron("intron-dev-public-3232.csv")

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

import torch
import pytorch_lightning as ptl

TOKENIZER_TYPE = "bpe" #@param ["bpe", "unigram"]

INTRON_VOCAB_SIZE = len(intron_train_set) + 2

os.system("python3 scripts/process_asr_text_tokenizer.py \
  --manifest=" + intron_train_manifest_cleaned + "\
  --vocab_size=" + str(INTRON_VOCAB_SIZE) + " \
  --data_root=" + tokenizer_dir + " \
  --tokenizer=\"spe\" \
  --spe_type=\"bpe\" \
  --spe_character_coverage=1.0 \
  --no_lower_case \
  --log")

TOKENIZER_DIR = f"{tokenizer_dir}/tokenizer_spe_{TOKENIZER_TYPE}_v{INTRON_VOCAB_SIZE}/"
print("Tokenizer directory :", TOKENIZER_DIR)

# Number of tokens in tokenizer - 
with open(os.path.join(TOKENIZER_DIR, 'tokenizer.vocab')) as f:
  tokens = f.readlines()

num_tokens = len(tokens)
print("Number of tokens : ", num_tokens)

if num_tokens < INTRON_VOCAB_SIZE:
    print(
        f"The text in this dataset is too small to construct a tokenizer "
        f"with vocab size = {INTRON_VOCAB_SIZE}. Current number of tokens = {num_tokens}. "
        f"Please reconstruct the tokenizer with fewer tokens"
    )

model = nemo_asr.models.ASRModel.from_pretrained("stt_en_conformer_ctc_large", map_location='cpu')

# Preserve the decoder parameters in case weight matching can be done later
pretrained_decoder = model.decoder.state_dict()

model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR, new_tokenizer_type="bpe")

# Insert preserved model weights if shapes match
if model.decoder.decoder_layers[0].weight.shape == pretrained_decoder['decoder_layers.0.weight'].shape:
    model.decoder.load_state_dict(pretrained_decoder)
    logging.info("Decoder shapes matched - restored weights from pre-trained model")
else:
    logging.info("\nDecoder shapes did not match - could not restore decoder weights from pre-trained model.")

freeze_encoder = True #@param ["False", "True"] {type:"raw"}
freeze_encoder = bool(freeze_encoder)

if freeze_encoder:
  model.encoder.freeze()
  model.encoder.apply(enable_bn_se)
  logging.info("Model encoder has been frozen")
else:
  model.encoder.unfreeze()
  logging.info("Model encoder has been un-frozen")

## Update config

cfg = copy.deepcopy(model.cfg)

### Setup tokenizer

# Setup new tokenizer
cfg.tokenizer.dir = TOKENIZER_DIR
cfg.tokenizer.type = "bpe"

# Set tokenizer config
model.cfg.tokenizer = cfg.tokenizer

### Setup data loaders

# Setup train/val/test configs
cfg.train_ds.is_tarred = False
print(OmegaConf.to_yaml(cfg.train_ds))

# Setup train, validation, test configs
with open_dict(cfg):
  # Train dataset
  cfg.train_ds.manifest_filepath = intron_train_manifest_cleaned
  cfg.train_ds.batch_size = 16
  cfg.train_ds.num_workers = 4
  cfg.train_ds.is_tarred: False # If set to true, uses the tarred version of the Dataset
  cfg.tarred_audio_filepaths: None
  cfg.train_ds.pin_memory = True
  cfg.train_ds.use_start_end_token = True
  cfg.train_ds.trim_silence = True

  # Validation dataset
  cfg.validation_ds.manifest_filepath = intron_dev_manifest_cleaned
  cfg.validation_ds.batch_size = 8
  cfg.validation_ds.num_workers = 4
  cfg.validation_ds.pin_memory = True
  cfg.validation_ds.use_start_end_token = True
  cfg.validation_ds.trim_silence = True

  # Test dataset
  cfg.test_ds.manifest_filepath = intron_dev_manifest_cleaned
  cfg.test_ds.batch_size = 8
  cfg.test_ds.num_workers = 4
  cfg.test_ds.pin_memory = True
  cfg.test_ds.use_start_end_token = True
  cfg.test_ds.trim_silence = True

# setup model with new configs
model.setup_training_data(cfg.train_ds)
model.setup_multiple_validation_data(cfg.validation_ds)
model.setup_multiple_test_data(cfg.test_ds)

def analyse_ctc_failures_in_model(model):
    count_ctc_failures = 0
    am_seq_lengths = []
    target_seq_lengths = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    mode = model.training
    
    train_dl = model.train_dataloader()

    with torch.no_grad():
      model = model.eval()
      for batch in tqdm(train_dl, desc='Checking for CTC failures'):
          x, x_len, y, y_len = batch
          x, x_len = x.to(device), x_len.to(device)
          x_logprobs, x_len, greedy_predictions = model(input_signal=x, input_signal_length=x_len)

          # Find how many CTC loss computation failures will occur
          for xl, yl in zip(x_len, y_len):
              if xl <= yl:
                  count_ctc_failures += 1

          # Record acoustic model lengths=
          am_seq_lengths.extend(x_len.to('cpu').numpy().tolist())

          # Record target sequence lengths
          target_seq_lengths.extend(y_len.to('cpu').numpy().tolist())
          
          del x, x_len, y, y_len, x_logprobs, greedy_predictions
    
    if mode:
      model = model.train()
      
    return count_ctc_failures, am_seq_lengths, target_seq_lengths

### Setup optimizer and scheduler

print(OmegaConf.to_yaml(cfg.optim))

##Reduce learning rate and warmup if required

with open_dict(model.cfg.optim):
  model.cfg.optim.lr = 0.025
  model.cfg.optim.weight_decay = 0.001
  model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
  model.cfg.optim.sched.warmup_ratio = 0.10  # 10 % warmup
  model.cfg.optim.sched.min_lr = 1e-9

### Setup data augmentation

with open_dict(model.cfg.spec_augment):
  model.cfg.spec_augment.freq_masks = 2
  model.cfg.spec_augment.freq_width = 25
  model.cfg.spec_augment.time_masks = 10
  model.cfg.spec_augment.time_width = 0.05

model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

## Setup Metrics

#@title Metric
use_cer = True #@param ["False", "True"] {type:"raw"}
log_prediction = True #@param ["False", "True"] {type:"raw"}

model._wer.use_cer = use_cer
model._wer.log_prediction = log_prediction

"""## Setup Trainer and Experiment Manager

And that's it! Now we can train the model by simply using the Pytorch Lightning Trainer and NeMo Experiment Manager as always.

For demonstration purposes, the number of epochs can be reduced. Reasonable results can be obtained in around 100 epochs (approximately 25 minutes on Colab GPUs).
"""

import torch
import pytorch_lightning as ptl

if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'cpu'

print(torch.cuda.is_available())

EPOCHS = 10  # 100 epochs would provide better results

trainer = ptl.Trainer(devices=1, 
                      accelerator=accelerator, 
                      max_epochs=EPOCHS, 
                      accumulate_grad_batches=4,
                      enable_checkpointing=False,
                      logger=False,
                      log_every_n_steps=5,
                      check_val_every_n_epoch=10)

# Setup model with the trainer
model.set_trainer(trainer)

# finally, update the model's internal config
model.cfg = model._cfg

from nemo.utils import exp_manager

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/lang-{LANGUAGE}/',
    name=f"ASR-Model-Language-{LANGUAGE}",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

config = OmegaConf.structured(config)

logdir = exp_manager.exp_manager(trainer, config)

trainer.fit(model)

# Save the final model

save_path = f"Model-{LANGUAGE}.nemo"
model.save_to(f"{save_path}")
print(f"Model saved at path : {os.getcwd() + os.path.sep + save_path}")