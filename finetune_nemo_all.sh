#!/bin/bash
#SBATCH --job-name=nemo_all
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_nemo_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_nemo_%j.txt

cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/
module load anaconda/3
module load cudatoolkit/11.7
conda activate /home/mila/c/chris.emezue/scratch/nemo3
#export LD_LIBRARY_PATH=/home/mila/c/chris.emezue/scratch/nemo2/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH


python3 src/train/nemo_ctc_finetuning.py --config src/config/nemo_all.ini
