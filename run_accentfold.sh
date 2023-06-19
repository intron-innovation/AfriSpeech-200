#!/bin/bash

# normal mode (1 and 2)
# CHEKPOINT_PATH='None' # default
# APPEND_NAME="_" # default
# EPOCH=10
# for B in "bini" "angas" "agatu"
# #for B in "angas"
# do
#     for neighbor in 10 20 35 10000 # 10000 corresponds to finetuning on the whole training dataset 
#     #for neighbor in 35
#     do
#         sbatch train_accent.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
#         #bash train_accent.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
#     done
# done



# #continuation of finetuning
CHEKPOINT_PATH="decipher" 
APPEND_NAME="_cont_6" # tells it to continue from 6 
EPOCH=10
for B in "bini" "angas" "agatu"
#for B in "agatu"
do
    for neighbor in 10 20 35
    #for neighbor in 20
    do
        #bash train_accent.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
    done
done


#bash train_accent.sh angas 35 _cont decipher 10