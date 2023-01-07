#!/bin/bash
#SBATCH --job-name=upload_whisper
#SBATCH --mem=20G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_upload_g_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_upload_g_%j.txt


cd ~/scratch/AfriSpeech-100/output
gdrive upload whisper_all/ --recursive
