import argparse
import configparser
import os
import subprocess
import time
from pathlib import Path

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/'
os.environ['XDG_CACHE_HOME'] = '/data/.cache/'
os.environ["WANDB_DISABLED"] = "true"

import torch
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    Wav2Vec2Tokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    is_apex_available,
    set_seed)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import warnings

from src.utils.prepare_dataset import DataConfig, data_prep, DataCollatorCTCWithPaddingGroupLen

warnings.filterwarnings('ignore')
wer_metric = load_metric("wer")
PROCESSOR = None


def parse_argument():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser(prog="Train")
    parser.add_argument("-c", "--config", dest="config_file",
                        help="Pass a training config file", metavar="FILE")
    parser.add_argument("--local_rank", type=int,
                        default=0)
    args = parser.parse_args()
    config.read(args.config_file)
    return args, config


def train_setup(config, args):
    exp_dir = config['experiment']['dir']
    checkpoints_path = config['checkpoints']['checkpoints_path']
    figure_path = config['logs']['figure_path']
    predictions_path = config['logs']['predictions_path']

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    subprocess.call(['cp', args.config_file, f"{exp_dir}/{args.config_file.split('/')[-1]}"])
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    Path(predictions_path).mkdir(parents=True, exist_ok=True)

    print(f"using exp_dir: {exp_dir}. Starting...")

    return checkpoints_path


def data_setup(config):
    data_config = DataConfig(
        train_path=config['data']['train'],
        val_path=config['data']['val'],
        exp_dir=config['experiment']['dir'],
        ckpt_path=config['checkpoints']['checkpoints_path'],
        model_path=config['models']['model_path'],
        audio_path=config['audio']['audio_path'],
        max_audio_len_secs=float(config['hyperparameters']['max_audio_len_secs']),
        min_transcript_len=float(config['hyperparameters']['min_transcript_len']),
        domain=config['data']['domain']
    )
    return data_config


def get_data_collator():
    return DataCollatorCTCWithPaddingGroupLen(processor=PROCESSOR, padding=True)


def compute_metric(pred):
    wer, _, _ = compute_wer(pred.predictions, pred.label_ids)
    return wer


def compute_wer(logits, label_ids):
    label_ids[label_ids == -100] = PROCESSOR.tokenizer.pad_token_id

    pred_ids = torch.argmax(torch.tensor(logits), axis=-1)
    predicted_transcription = PROCESSOR.batch_decode(pred_ids)[0]

    text = PROCESSOR.batch_decode(label_ids, group_tokens=False)[0]
    target_transcription = text.lower()

    wer = wer_metric.compute(predictions=[predicted_transcription],
                             references=[target_transcription])
    return {"wer": wer}, target_transcription, predicted_transcription


def get_checkpoint(checkpoint_path, model_path):
    last_checkpoint_ = None
    if os.path.isdir(checkpoint_path):
        last_checkpoint_ = get_last_checkpoint(checkpoint_path)
        if last_checkpoint_ is None and len(os.listdir(checkpoint_path)) > 0:
            print(
                f"Output directory ({checkpoint_path}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint_ is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint_}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # use last checkpoint if exist
    if last_checkpoint_:
        checkpoint = last_checkpoint_
    elif os.path.isdir(model_path):
        checkpoint = None
    else:
        checkpoint = None

    return last_checkpoint_, checkpoint


if __name__ == "__main__":

    args, config = parse_argument()
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)
    train_dataset, val_dataset, PROCESSOR = data_prep(data_config)
    data_collator = get_data_collator()

    start = time.time()
    # Detecting last checkpoint.
    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])

    CTC_model_class = Wav2Vec2ForCTC if 'hubert' not in config['models']['model_path'] else HubertForCTC

    models_with_different_vocab = ['jonatasgrosman/wav2vec2-large-xlsr-53-english',
                                   'facebook/wav2vec2-large-960h-lv60-self',
                                   'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200'
                                   ]

    print(f"model starting...from last checkpoint:{last_checkpoint}")
    if config['models']['model_path'] in models_with_different_vocab:
        from transformers.file_utils import hf_bucket_url, cached_path

        archive_file = hf_bucket_url(
            config['models']['model_path'],
            filename='pytorch_model.bin'
        )
        resolved_archive_file = cached_path(archive_file)

        state_dict = torch.load(resolved_archive_file, map_location='cpu')
        state_dict.pop('lm_head.weight')
        state_dict.pop('lm_head.bias')

        model = CTC_model_class.from_pretrained(
            config['models']['model_path'],
            state_dict=state_dict,
            attention_dropout=float(config['hyperparameters']['attention_dropout']),
            hidden_dropout=float(config['hyperparameters']['hidden_dropout']),
            feat_proj_dropout=float(config['hyperparameters']['feat_proj_dropout']),
            mask_time_prob=float(config['hyperparameters']['mask_time_prob']),
            layerdrop=float(config['hyperparameters']['layerdrop']),
            ctc_loss_reduction=config['hyperparameters']['ctc_loss_reduction'],
            ctc_zero_infinity=True,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id,
            vocab_size=len(PROCESSOR.tokenizer)
        )

    else:
        model = CTC_model_class.from_pretrained(
            last_checkpoint if last_checkpoint else config['models']['model_path'],
            attention_dropout=float(config['hyperparameters']['attention_dropout']),
            hidden_dropout=float(config['hyperparameters']['hidden_dropout']),
            feat_proj_dropout=float(config['hyperparameters']['feat_proj_dropout']),
            mask_time_prob=float(config['hyperparameters']['mask_time_prob']),
            layerdrop=float(config['hyperparameters']['layerdrop']),
            ctc_loss_reduction=config['hyperparameters']['ctc_loss_reduction'],
            ctc_zero_infinity=True,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id,
            vocab_size=len(PROCESSOR.tokenizer)
        )
    if config['hyperparameters']['gradient_checkpointing'] == "True":
        model.gradient_checkpointing_enable()

    if config['hyperparameters']['ctc_zero_infinity'] == "True":
        model.config.ctc_zero_infinity = True

    print(f"\n...Model loaded in {time.time() - start:.4f}.\n")

    if config['hyperparameters']['freeze_feature_encoder'] == "True":
        model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        overwrite_output_dir=True if config['hyperparameters']['overwrite_output_dir'] == "True" else False,
        group_by_length=True if config['hyperparameters']['group_by_length'] == "True" else False,
        length_column_name=config['hyperparameters']['length_column_name'],
        data_seed=int(config['hyperparameters']['data_seed']),
        per_device_train_batch_size=int(config['hyperparameters']['train_batch_size']),
        per_device_eval_batch_size=int(config['hyperparameters']['val_batch_size']),
        gradient_accumulation_steps=int(config['hyperparameters']['gradient_accumulation_steps']),
        gradient_checkpointing=True if config['hyperparameters']['gradient_checkpointing'] == "True" else False,
        ddp_find_unused_parameters=True if config['hyperparameters']['ddp_find_unused_parameters'] == "True" else False,
        evaluation_strategy="steps",
        num_train_epochs=int(config['hyperparameters']['num_epochs']),
        fp16=torch.cuda.is_available(),
        save_steps=int(config['hyperparameters']['save_steps']),
        eval_steps=int(config['hyperparameters']['eval_steps']),
        logging_steps=int(config['hyperparameters']['logging_steps']),
        learning_rate=float(config['hyperparameters']['learning_rate']),
        warmup_steps=int(config['hyperparameters']['warmup_steps']),
        save_total_limit=int(config['hyperparameters']['save_total_limit']),
        dataloader_num_workers=int(config['hyperparameters']['dataloader_num_workers']),
        logging_first_step=True,
        load_best_model_at_end=True if config['hyperparameters']['load_best_model_at_end'] == 'True' else False,
        metric_for_best_model='eval_wer',
        greater_is_better=False,
        ignore_data_skip=True if config['hyperparameters']['ignore_data_skip'] == 'True' else False,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metric,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=PROCESSOR.feature_extractor,
    )

    PROCESSOR.save_pretrained(checkpoints_path)

    print(f"\n...Model Args loaded in {time.time() - start:.4f}. Start training...\n")

    trainer.train(resume_from_checkpoint=checkpoint_)

    model.save_pretrained(checkpoints_path)
    PROCESSOR.save_pretrained(checkpoints_path)
