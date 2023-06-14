#!/bin/bash

# normal mode (1 and 2)
CHEKPOINT_PATH='None' # default
APPEND_NAME="_" # default
EPOCH=10
for B in "bini" "angas" "agatu"
#for B in "angas"
do
    for neighbor in 10 20 35 10000 # 10000 corresponds to finetuning on the whole training dataset 
    #for neighbor in 35
    do
        sbatch train_accent.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
        #bash train_accent.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
    done
done



# continuation of finetuning
#CHEKPOINT_PATH='/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4/wav2vec2-large-xlsr-53-accentfold_angas_35_/checkpoints/checkpoint-3' # default
#APPEND_NAME="_cont" # default
#EPOCH=1

#/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4/wav2vec2-large-xlsr-53-accentfold_angas_35_/checkpoints/checkpoint-3

#bash train_accent.sh $B $neighbor _cont checkpoint epoch #         when we finetune on D and continue on A

