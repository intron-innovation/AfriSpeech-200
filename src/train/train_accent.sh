#!/bin/bash
cd /home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper
module load anaconda/3
#module load cuda/11.0/cudnn/8.0
conda activate /home/mila/c/chris.emezue/scratch/nemo2
export PYTHONPATH=$PYTHONPATH:/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper

python3 -m src.train.train -c src/config/accent_fold/config_xlsr_accentfold.ini