#!/bin/bash
#SBATCH --array=0-107
#SBATCH --job-name=accent-fold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurm_arr/slurmerror_accentfold_%A_%a.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurm_arr/slurmoutput_accentfold_%A_%a.txt



cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cudatoolkit/11.7
conda activate /home/mila/c/chris.emezue/scratch/afrispeech

#export SLURM_ARRAY_TASK_ID='bini' # in debug mode

python3 -m src.train.train \
-c src/config/accent_fold/config_xlsr_accentfold_chris_new_idea.ini \
--experiment_name wav2vec2-large-xlsr-53-accentfold_${SLURM_ARRAY_TASK_ID}_${2}${3} \
-k $2 \
-b $SLURM_ARRAY_TASK_ID \
--epoch $5 \
--checkpointPath $4 