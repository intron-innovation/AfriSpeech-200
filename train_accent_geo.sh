#!/bin/bash
#SBATCH --job-name=accent-fold-geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurm_geo/slurmerror_accentfold_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurm_geo/slurmoutput_accentfold_%j.txt



cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cudatoolkit/11.7
conda activate /home/mila/c/chris.emezue/scratch/afrispeech

python3 -m src.train.train \
-c src/config/accent_fold/config_xlsr_accentfold_chris_geo_idea.ini \
--experiment_name wav2vec2-large-xlsr-53-accentfold_"${1}"_${2}${3} \
-k $2 \
-b "$1" \
--epoch $5 \
--checkpointPath $4



#-c src/config/accent_fold/config_xlsr_accentfold_chris_m1.ini \
