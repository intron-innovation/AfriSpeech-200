#!/bin/bash
#SBATCH --job-name=finetune_whisper_all
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_all_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_all_%j.txt

cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cuda/10.0/cudnn/7.6
conda activate /home/mila/c/chris.emezue/scratch/afrispeech


python3 src/train/whisper_finetuning.py -c src/config/whisper_all.ini
#python src/experiments/whisper_finetuning.py \
#--evaluate \
#--train \
#--audio_dir=/home/mila/c/chris.emezue/scratch/AfriSpeech-100/ \
#--output_dir=/home/mila/c/chris.emezue/scratch/AfriSpeech-100/output