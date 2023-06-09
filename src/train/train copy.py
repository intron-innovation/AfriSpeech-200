import os

data_home = "data3"
os.environ['TRANSFORMERS_CACHE'] = f'/{data_home}/.cache/'
os.environ['XDG_CACHE_HOME'] = f'/{data_home}/.cache/'
os.environ["WANDB_DISABLED"] = "true"

import argparse
import configparser
import random
import subprocess
import time
import warnings
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
import evaluate
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    TrainingArguments,
    Trainer,
    )
from transformers.trainer_utils import get_last_checkpoint

from src.utils.text_processing import clean_text, strip_task_tags
from src.utils.prepare_dataset import DataConfig, data_prep, DataCollatorCTCWithPaddingGroupLen, DISCRIMINATIVE
from src.utils.sampler import IntronTrainer
from src.utils.compute_cluster_distance import compute_distances
from src.train.models import Wav2Vec2ForCTCnCLS

warnings.filterwarnings('ignore')
wer_metric = load_metric("wer")
SAMPLING_RATE = 16000
PROCESSOR = None

num_of_gpus = torch.cuda.device_count()
print("num_of_gpus:", num_of_gpus)
print("torch.cuda.is_available()", torch.cuda.is_available())
#device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
print("cuda.current_device:", torch.cuda.current_device())
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


def parse_argument():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser(prog="Train")
    parser.add_argument("-c", "--config", dest="config_file",
                        help="Pass a training config file", metavar="FILE")
    parser.add_argument("--local_rank", type=int,
                        default=0)
    parser.add_argument("-g", "-gpu", "--gpu", type=int,
                        default=0)
    args = parser.parse_args()
    config.read(args.config_file)
    return args, config


def train_setup(config, args):
    repo_root = config['experiment']['repo_root']
    exp_dir = os.path.join(repo_root, config['experiment']['dir'], config['experiment']['name'])
    config['experiment']['dir'] = exp_dir
    checkpoints_path = os.path.join(exp_dir, 'checkpoints')
    config['checkpoints']['checkpoints_path'] = checkpoints_path
    figure_path = os.path.join(exp_dir, 'figures')
    config['logs']['figure_path'] = figure_path
    predictions_path = os.path.join(exp_dir, 'predictions')
    config['logs']['predictions_path'] = predictions_path

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    subprocess.call(['cp', args.config_file, f"{exp_dir}/{args.config_file.split('/')[-1]}"])
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    Path(predictions_path).mkdir(parents=True, exist_ok=True)

    print(f"using exp_dir: {exp_dir}. Starting...")

    return checkpoints_path


def data_setup(config):
    multi_task = {}
    if 'tasks' in config:
        multi_task['architecture'] = config['tasks']['architecture'] if 'architecture' in config['tasks'] else None
        multi_task['expand_vocab'] = True if config['tasks']['expand_vocab'] == "True" else False
        multi_task['expand_vocab_mode'] = config['tasks']['expand_mode']
        multi_task['accent'] = True if config['tasks']['accent'] == "True" else False
        multi_task['domain'] = True if config['tasks']['domain'] == "True" else False
        multi_task['vad'] = True if config['tasks']['vad'] == "True" else False

    data_config = DataConfig(
        train_path=config['data']['train'],
        val_path=config['data']['val'],
        test_path=config['data']['test'],
        aug_path=config['data']['aug'] if 'aug' in config['data'] else None,
        aug_percent=float(config['data']['aug_percent']) if 'aug_percent' in config['data'] else None,
        exp_dir=config['experiment']['dir'],
        ckpt_path=config['checkpoints']['checkpoints_path'],
        model_path=config['models']['model_path'],
        audio_path=config['audio']['audio_path'],
        max_audio_len_secs=int(config['hyperparameters']['max_audio_len_secs']),
        min_transcript_len=int(config['hyperparameters']['min_transcript_len']),
        domain=config['data']['domain'],
        seed=int(config['hyperparameters']['data_seed']),
        multi_task=multi_task,
        accent_subset = list(config['hyperparameters']['accent_subset'])
    )
    return data_config


def get_data_collator(multi_task):
    data_collator_ = DataCollatorCTCWithPaddingGroupLen(processor=PROCESSOR, padding=True)
    data_collator_.multi_task=multi_task
    return data_collator_


def compute_metric(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = PROCESSOR.tokenizer.pad_token_id

    pred_str_list = PROCESSOR.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str_list = PROCESSOR.batch_decode(pred.label_ids, group_tokens=False)
    
    pred_str_list = [strip_task_tags(text) for text in pred_str_list]
    label_str_list = [strip_task_tags(text) for text in label_str_list]

    wer = wer_metric.compute(predictions=pred_str_list,
                             references=label_str_list)

    return {"wer": wer}


def get_checkpoint(checkpoint_path, model_path):
    last_checkpoint_ = None
    
    ckpt_files = os.listdir(checkpoint_path)
    if "pytorch_model.bin" in ckpt_files:
        return checkpoint_path, checkpoint_path
    
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


def set_dropout(trained_model):
    trained_model.eval()
    for name, module in trained_model.named_modules():
        if 'dropout' in name:
            module.train()


def run_inference(trained_model, dataloader, mode='most', mc_dropout_rounds=10):
    if mode == 'random':
        # we do not need to compute the WER here
        # we shuffle randomly the dictionary (this will display a random order) - the selecting the strict first top-k
        audios_ids = [batch['audio_idx'] for batch in dataloader]
        random.shuffle(audios_ids)
        return {key: 1.0 for key in
                audios_ids}  # these values are just dummy ones, to have a format similar to the two other cases
    else:
        audio_wers = {}
        for batch in tqdm(dataloader, desc="AL inference"):
            input_val = batch['input_values'].to(device)

            # run 10 steps of mc dropout
            wer_list = []
            batch["reference"] = clean_text(PROCESSOR.batch_decode(batch["labels"])[0])
            
            for mc_dropout_round in range(mc_dropout_rounds):
                with torch.no_grad():
                    logits = trained_model(input_val).logits
                    batch["logits"] = logits

                pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
                pred = PROCESSOR.batch_decode(pred_ids)[0]
                batch["predictions"] = clean_text(pred)
                batch["wer"] = wer_metric.compute(
                    predictions=[batch["predictions"]], references=[batch["reference"]]
                )
                wer_list.append(batch['wer'])
            uncertainty_score = np.array(wer_list).std()
            audio_wers[batch['audio_idx'][0]] = uncertainty_score
        
        if mode == 'most':
            # we select most uncertain samples
            return dict(sorted(audio_wers.items(), key=lambda item: item[1]), reverse=True)
        if mode == 'least':
            # we select the least uncertain samples
            return dict(sorted(audio_wers.items(), key=lambda item: item[1]), reverse=False)
        raise NotImplementedError


if __name__ == "__main__":

    args, config = parse_argument()
    # print("args.gpu:", args.gpu)
    # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(args.gpu)
    # print("cuda.current_device:", torch.cuda.current_device())
    

    accent_B = 'twi'
    k_accents = 10
    ##computing centroid.
    centroids = pd.read_csv("./data/afrispeech_accents_centroids.csv").set_index("accent")
    #use euclidean distance 
    accent_subset = compute_distances(list(centroids[accent_B]), centroids, k_accents)

    config['hyperparameters']['accent_subset'] = accent_subset
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)
    A_train_dataset, D_train_dataset B_test_dataset, aug_dataset, PROCESSOR = data_prep(data_config)
    data_collator = get_data_collator(data_config.multi_task)

    start = time.time()
    # Detecting last checkpoint.
    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])

    CTC_model_class = None
    if 'hubert' in config['models']['model_path']:
        CTC_model_class = HubertForCTC
    elif 'tasks' in config and config['tasks']['architecture'] == DISCRIMINATIVE:
        CTC_model_class = Wav2Vec2ForCTCnCLS
    else:
        CTC_model_class = Wav2Vec2ForCTC

    models_with_different_vocab = ['jonatasgrosman/wav2vec2-large-xlsr-53-english',
                                   'facebook/wav2vec2-large-960h-lv60-self',
                                   'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200'
                                   ]

    print(f"model starting...from last checkpoint:{last_checkpoint}")
    
    if 'tasks' in config and config['tasks']['architecture'] == DISCRIMINATIVE:
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
            vocab_size=len(PROCESSOR.tokenizer),
            accent=True if config['tasks']['accent'] == "True" else False,
            domain=True if config['tasks']['domain'] == "True" else False,
            vad=True if config['tasks']['vad'] == "True" else False,
            loss_reduction=config['tasks']['loss_reduction'],
            alphas=config['tasks']['alphas'],
            accent_len=int(config['tasks']['num_accents']), 
            domain_len=int(config['tasks']['num_domains']), 
            vad_len=int(config['tasks']['num_vad'])
        )
    elif config['models']['model_path'] in models_with_different_vocab:
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
            vocab_size=len(PROCESSOR.tokenizer),
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
        evaluation_strategy="epochs",
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
        report_to=None
    )
    
    print("device: ", training_args.device, device)
    
    print(f"\n...Model Args loaded in {time.time() - start:.4f}. Start training...\n")

    trainer = IntronTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metric,
        train_dataset=A_train_dataset,
        tokenizer=PROCESSOR.feature_extractor,
        sampler=config['data']['sampler'] if 'sampler' in config['data'] else None
    )
    
    if config['hyperparameters']['do_train'] == "True":
        PROCESSOR.save_pretrained(checkpoints_path)
        
        trainer.train(resume_from_checkpoint=checkpoint_)

        model.save_pretrained(checkpoints_path)
        PROCESSOR.save_pretrained(checkpoints_path)

    if config['hyperparameters']['do_eval'] == "True":
        metrics = Atrainer.evaluate(B_test_dataset)
        metrics["eval_samples"] = len(val_dataset)
        metrics["eval_samples"] = str(f"{accent_B}_{str(k_accents)}")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    