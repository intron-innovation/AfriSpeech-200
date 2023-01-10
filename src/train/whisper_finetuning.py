#
# Code mostly borrowed from https://huggingface.co/blog/fine-tune-whisper 🤗

# Other sources
# https://huggingface.co/sanchit-gandhi/whisper-medium-switchboard-5k/blob/main/run_speech_recognition_whisper.py
# https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz
#

import os

#os.environ['TRANSFORMERS_CACHE'] = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/results/'
#os.environ['XDG_CACHE_HOME'] = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/results/'
os.environ["WANDB_DISABLED"] = "true"

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
    Seq2SeqTrainer,
)
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.train.train import parse_argument, train_setup, get_checkpoint, data_setup, set_dropout
from src.inference.inference import write_pred
from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_custom_dataset
from src.utils.text_processing import clean_text


set_seed(1778)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_audio = '/data/data/intron/e809b58c-4f05-4754-b98c-fbf236a88fbc/544bbfe5e1c6f8afb80c4840b681908d.wav'

num_gpus = [i for i in range(torch.cuda.device_count())]
if len(num_gpus) > 1:
    print("Let's use", num_gpus, "GPUs!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)
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
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

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

def get_whisper_pretrained_model(checkpoint_pretrained, config_):

    model_ = WhisperForConditionalGeneration.from_pretrained(
        checkpoint_pretrained if checkpoint_pretrained else config_['models']['model_path'],
    )
    if len(num_gpus) > 1:
        model_ = torch.nn.DataParallel(model_, device_ids=num_gpus)
    model_.to(device)
    return model_.module if len(num_gpus) > 1 else model_

def run_whisper_inference(trained_model, dataloader, mode='most', mc_dropout_rounds=10):
    # print('In inference:', type(mode))
    if 'random' in mode.lower():
        # print('Gets in random')
        # we do not need to compute the WER here
        # we shuffle randomly the dictionary (this will display a random order) - the selecting the strict first top-k
        audios_ids = [batch['audio_idx'] for batch in dataloader]
        random.shuffle(audios_ids)
        return {key[0]: 1.0 for key in
                audios_ids}  # these values are just dummy ones, to have a format similar to the two other cases
    else:
        audio_wers = {}
        final_dict = {}
        for batch in tqdm(dataloader, desc="Uncertainty Inference"):
            input_val = batch['input_values'].to(device)

            # run 10 steps of mc dropout
            wer_list = []
            batch["reference"] = clean_text(processor.batch_decode(batch["labels"])[0])

            for mc_dropout_round in range(mc_dropout_rounds):
                with torch.no_grad():
                    logits = trained_model(input_val).logits
                    batch["logits"] = logits

                pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
                pred = processor.batch_decode(pred_ids)[0]
                batch["predictions"] = clean_text(pred)
                # the following block is against cases where we have empty reference
                # leading to the error: "ValueError: one or more groundtruths are empty strings"
                try:
                    batch["wer"] = wer_metric.compute(
                        predictions=[batch["predictions"]], references=[batch["reference"]]
                    )
                    wer_list.append(batch['wer'])
                except:
                    pass

            if len(wer_list) > 0:
                uncertainty_score = np.array(wer_list).std()
                audio_wers[batch['audio_idx'][0]] = uncertainty_score

        if 'most' in mode.lower():
            # we select most uncertain samples
            return dict(sorted(audio_wers.items(), key=lambda item: item[1], reverse=True))
        if 'least' in mode.lower():
            # we select the least uncertain samples
            return dict(sorted(audio_wers.items(), key=lambda item: item[1], reverse=False))

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
            print(f"{audio_path} not found {str(e)}")
            speech = load_audio_file(temp_audio)
        
        # Compute log-Mel input features from input audio array
        audio = feature_extractor(speech, sampling_rate=AudioConfig.sr).input_features[0]

        # Encode target text to label ids
        text = clean_text(text)
        labels = tokenizer(text.lower()).input_ids

        return audio, labels

    # Load the dataset
    dev_dataset = load_custom_dataset(data_config, data_config.val_path, 
                                      'dev', transform_dataset, prepare=True)
    train_dataset = load_custom_dataset(data_config, data_config.train_path, 
                                        'train', transform_dataset, prepare=True)
    aug_dataset = load_custom_dataset(data_config, data_config.aug_path, 
                                        'aug', transform_dataset, prepare=True)

    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])
    print(f"model starting...from last checkpoint:{last_checkpoint}")

    # load model
    model = get_whisper_pretrained_model(last_checkpoint, config)
    
    if config['hyperparameters']['do_train'] == "True":

        # Override generation arguments
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

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
            report_to=config['hyperparameters']['report_to'],
            load_best_model_at_end=True if config['hyperparameters']['load_best_model_at_end'] == 'True' else False,
            metric_for_best_model='eval_wer',
            greater_is_better=False,
            push_to_hub=False,
            logging_first_step=True,
            dataloader_num_workers=int(config['hyperparameters']['dataloader_num_workers']),
        )

        # # Define the trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        processor.save_pretrained(checkpoints_path)
        trainer.train(resume_from_checkpoint=checkpoint_)
        model.save_pretrained(checkpoints_path)
        processor.save_pretrained(checkpoints_path)

        if 'aug' in config['data']:
            # after baseline is completed

            print(f"\n...Baseline model trained in {time.time() - start:.4f}. Start training with Active Learning...\n")

            active_learning_rounds = int(config['hyperparameters']['active_learning_rounds'])
            aug_batch_size = int(config['hyperparameters']['aug_batch_size'])
            sampling_mode = str(config['hyperparameters']['sampling_mode']).strip()
            k = int(config['hyperparameters']['top_k'])
            mc_dropout_round = int(config['hyperparameters']['mc_dropout_round'])

            # AL rounds
            for active_learning_round in range(active_learning_rounds):
                print('Performing McDropout for AL Round: {}\n'.format(active_learning_round))

                # McDropout for uncertainty computation
                set_dropout(model)
                # evaluation step and uncertain samples selection
                augmentation_dataloader = DataLoader(aug_dataset, batch_size=aug_batch_size)

                samples_uncertainty = run_whisper_inference(model, augmentation_dataloader,
                                                    mode=sampling_mode, mc_dropout_rounds=mc_dropout_round)
                uncertainties = np.array(list(samples_uncertainty.values()))
                min_uncertainty = uncertainties.min()
                max_uncertainty = uncertainties.max()
                mean_uncertainty = uncertainties.mean()
                print('AL Round: {} with SM: {} - Max Uncertainty: {} - Min Uncertainty: {} - Mean Uncertainty: {}'.format(active_learning_round,
                                                                                                    sampling_mode,
                                                                                                    max_uncertainty,
                                                                                                    min_uncertainty, mean_uncertainty))
                # top-k samples
                most_uncertain_samples_idx = list(samples_uncertainty.keys())[:k]

                # writing the top=k to disk
                filename = 'Top-{}_AL_Round_{}_Mode_{}'.format(k, active_learning_round, sampling_mode)
                # write the top-k to the disk
                filepath = os.path.join(checkpoints_path, filename)
                np.save(filepath, np.array(most_uncertain_samples_idx + [max_uncertainty, min_uncertainty, mean_uncertainty])) # appending uncertainties stats to keep track
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

                # delete current model from memory and empty cache
                del model

                torch.cuda.empty_cache()

                if len(aug_dataset) == 0 or len(aug_dataset) < k:
                    print('Stopping AL because the augmentation dataset is now empty or less than top-k ({})'.format(k))
                    break
                else:
                    model = get_whisper_pretrained_model(last_checkpoint, config)
                    # reset the trainer with the updated training and augmenting dataset
                    new_al_round_checkpoint_path = os.path.join(checkpoints_path, f"AL_Round_{active_learning_round+1}")
                    Path(new_al_round_checkpoint_path).mkdir(parents=True, exist_ok=True)

                    # Detecting last checkpoint.
                    last_checkpoint, checkpoint_ = get_checkpoint(new_al_round_checkpoint_path,
                                                                  config['models']['model_path'])
                    # update training arg with new output path
                    training_args.output_dir = new_al_round_checkpoint_path

                    trainer = Seq2SeqTrainer(
                        args=training_args,
                        model=model,
                        train_dataset=train_dataset,
                        eval_dataset=dev_dataset,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        tokenizer=processor.feature_extractor,
                    )

                    processor.save_pretrained(new_al_round_checkpoint_path)
                    print('Active Learning Round: {}\n'.format(active_learning_round+1))
                    trainer.train(resume_from_checkpoint=checkpoint_)
                    # define path for checkpoints for new AL round
                    model.module.save_pretrained(new_al_round_checkpoint_path)
