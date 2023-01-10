#!/bin/bash
#SBATCH --job-name=al_whisper
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/afrispeech_al/AfriSpeech-Dataset-Paper/slurmerror_general_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/afrispeech_al/AfriSpeech-Dataset-Paper/slurmoutput_general_%j.txt

cd /home/mila/c/chris.emezue/afrispeech_al/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cuda/10.0/cudnn/7.6
conda activate /home/mila/c/chris.emezue/scratch/afrispeech


python3 -m pdb src/train/train.py -c src/config/config_al_xlsr_general.ini
