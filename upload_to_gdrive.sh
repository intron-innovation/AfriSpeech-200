#!/bin/bash
#SBATCH --job-name=upload_nemo
#SBATCH --mem=20G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_nemo_upload_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_nemo_upload_%j.txt

cd ~/scratch/AfriSpeech-100/output
gdrive upload nemo_experiments/ --recursive
