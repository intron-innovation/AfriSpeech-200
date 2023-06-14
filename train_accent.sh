#!/bin/bash
#SBATCH --job-name=accent-fold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper$/slurmerror_accentfold_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper$/slurmoutput_accentfold_%j.txt



#cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/src/train
#module load anaconda/3
#module load cuda/11.0/cudnn/8.0
#conda activate /home/mila/c/chris.emezue/scratch/nemo2
#export PYTHONPATH=$PYTHONPATH:/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper

cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cudatoolkit/11.7
conda activate /home/mila/c/chris.emezue/scratch/afrispeech

python3 -m src.train.train -c src/config/accent_fold/config_xlsr_accentfold_chris.ini -k $2 -b $1