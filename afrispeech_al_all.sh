#!/bin/bash
#SBATCH --job-name=afrispeech_al
#SBATCH --gres=gpu:rtx8000:3
#SBATCH --cpus-per-gpu=18
#SBATCH --mem=180G
#SBATCH --time=168:00:00         
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/comError.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/comOutput.txt


###########cluster information above this line
module load python/3.9 cuda/10.2/cudnn/7.6
source source /home/mila/b/bonaventure.dossou/afrispeech/bin/activate
export PYTHONPATH=$PYTHONPATH:~/AfriSpeech-Dataset-Paper
python src/train/train.py -c src/config/config_al_xlsr_general.ini
python src/train/train.py -c src/config/config_al_xlsr_clinical.ini
python src/train/train.py -c src/config/config_al_xlsr.ini