import os
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

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/'
os.environ['XDG_CACHE_HOME'] = '/data/.cache/'
os.environ["WANDB_DISABLED"] = "true"

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    TrainingArguments,
    Trainer,
    )
from transformers.trainer_utils import get_last_checkpoint

from src.utils.text_processing import clean_text
from src.utils.prepare_dataset import DataConfig, data_prep, DataCollatorCTCWithPaddingGroupLen
from src.utils.sampler import IntronTrainer

warnings.filterwarnings('ignore')
wer_metric = load_metric("wer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000
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
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)
    train_dataset, val_dataset, aug_dataset, PROCESSOR = data_prep(data_config)
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
        report_to=None
    )
    
    print(f"\n...Model Args loaded in {time.time() - start:.4f}. Start training...\n")

    trainer = IntronTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metric,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=PROCESSOR.feature_extractor,
        sampler=config['data']['sampler'] if 'sampler' in config['data'] else None
    )

    PROCESSOR.save_pretrained(checkpoints_path)

    trainer.train(resume_from_checkpoint=checkpoint_)

    model.save_pretrained(checkpoints_path)
    PROCESSOR.save_pretrained(checkpoints_path)

    if 'aug' in config['data']:
        # after baseline is completed
        
        print(f"\n...Baseline model trained in {time.time() - start:.4f}. Start training with Active Learning...\n")

        active_learning_rounds = int(config['hyperparameters']['active_learning_rounds'])
        aug_batch_size = int(config['hyperparameters']['aug_batch_size'])
        sampling_mode = config['hyperparameters']['sampling_mode']
        k = float(config['hyperparameters']['top_k'])
        if k < 1:
            k = len(aug_dataset)/active_learning_rounds
        k = int(k)
        mc_dropout_round = int(config['hyperparameters']['mc_dropout_round'])

        # AL rounds
        for active_learning_round in range(active_learning_rounds):
            print('Active Learning Round: {}\n'.format(active_learning_round))

            # McDropout for uncertainty computation
            set_dropout(model)
            # evaluation step and uncertain samples selection
            augmentation_dataloader = DataLoader(aug_dataset, batch_size=aug_batch_size)

            samples_uncertainty = run_inference(model, augmentation_dataloader,
                                                mode=sampling_mode, mc_dropout_rounds=mc_dropout_round)
            # top-k samples (select top-3k)
            most_uncertain_samples_idx = list(samples_uncertainty.keys())[:k]

            # writing the top=k to disk
            filename = 'Top-{}_AL_Round_{}_Mode_{}'.format(k, active_learning_round, sampling_mode)
            # write the top-k to the disk
            filepath = os.path.join(checkpoints_path, filename)
            np.save(filepath, np.array(most_uncertain_samples_idx))
            print(f"saved audio ids for round {active_learning_round} to {filepath}")

            print('Old training set size: {} - Old Augmenting Size: {}'.format(len(train_dataset), len(aug_dataset)))
            augmentation_data = aug_dataset.get_dataset()
            training_data = train_dataset.get_dataset()
            # get top-k samples of the augmentation set
            selected_samples_df = augmentation_data[augmentation_data.audio_ids.isin(most_uncertain_samples_idx)]
            # remove those samples from the augmenting set and set the new augmentation set
            new_augmenting_samples = augmentation_data[~augmentation_data.audio_ids.isin(most_uncertain_samples_idx)]
            aug_dataset.set_dataset(new_augmenting_samples)
            # add the new dataset to the training set
            new_training_data = pd.concat([training_data, selected_samples_df])
            train_dataset.set_dataset(new_training_data)
            print('New training set size: {} - New Augmenting Size: {}'.format(len(train_dataset), len(aug_dataset)))

            # set model back to eval before training mode
            model.eval()

            # reset the trainer with the updated training and augmenting dataset
            new_al_round_checkpoint_path = os.path.join(checkpoints_path, f"AL_Round_{active_learning_round}")
            Path(new_al_round_checkpoint_path).mkdir(parents=True, exist_ok=True)

            # Detecting last checkpoint.
            last_checkpoint, checkpoint_ = get_checkpoint(new_al_round_checkpoint_path,
                                                          config['models']['model_path'])
            # update training arg with new output path
            training_args.output_dir = new_al_round_checkpoint_path

            trainer = IntronTrainer(
                model=model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=compute_metric,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=PROCESSOR.feature_extractor,
            )
            PROCESSOR.save_pretrained(new_al_round_checkpoint_path)

            trainer.train(resume_from_checkpoint=checkpoint_)

            # define path for checkpoints for new AL round
            model.save_pretrained(new_al_round_checkpoint_path)

