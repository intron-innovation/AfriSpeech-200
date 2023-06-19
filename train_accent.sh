#!/bin/bash
#SBATCH --job-name=accent-fold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/afrispeech-multitask/AfriSpeech-Dataset-Paper/slurm/slurmerror_accentfold_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/afrispeech-multitask/AfriSpeech-Dataset-Paper/slurm/slurmoutput_accentfold_%j.txt



cd /home/mila/c/chris.emezue/afrispeech-multitask/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cudatoolkit/11.7
conda activate /home/mila/c/chris.emezue/scratch/afrispeech

python3 -m src.train.train \
-c src/config/config_xlsr_3_heads_asr_accent_domain_weighted_5_3_2_chris.ini