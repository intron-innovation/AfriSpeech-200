#!/bin/bash
#SBATCH --job-name=nemo_al
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_nemo_inference_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_nemo_inference_%j.txt

cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
#module load cuda/11.0/cudnn/8.0
conda activate /home/mila/c/chris.emezue/scratch/nemo2
#export PYTHONPATH=$PYTHONPATH:/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
#export MKL_SERVICE_FORCE_INTEL=1
#export LD_LIBRARY_PATH=/home/mila/c/chris.emezue/scratch/nemo2/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

python3 mlm_sense.py
