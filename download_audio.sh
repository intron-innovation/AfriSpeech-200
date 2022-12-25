#!/bin/bash
#SBATCH --job-name=aws_audio
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmerror_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/slurmoutput_%j.txt




cd /home/mila/c/chris.emezue/scratch/AfriSpeech-100
aws s3 cp s3://intron-open-source/AfriSpeech-100 . --recursive