#!/bin/bash
#SBATCH --job-name=finetune_whisper
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_nemo_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_nemo_%j.txt

cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cuda/11.0/cudnn/8.0
conda activate /home/mila/c/chris.emezue/scratch/nemo2


python3 -m pdb src/train/nemo_ctc_finetuning.py --config src/config/nemo_all.ini
#python3 src/train/nemo_ctc_finetuning.py --config src/config/nemo_all.ini
