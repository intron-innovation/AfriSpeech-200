#!/bin/bash
#cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/src/train
#module load anaconda/3
#module load cuda/11.0/cudnn/8.0
#conda activate /home/mila/c/chris.emezue/scratch/nemo2
#export PYTHONPATH=$PYTHONPATH:/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper

cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
module load cudatoolkit/11.7
#module load cuda/10.0/cudnn/7.6
conda activate /home/mila/c/chris.emezue/scratch/afrispeech

python3 -m src.train.train -c src/config/accent_fold/config_xlsr_accentfold_chris.ini -k 15 -b akan