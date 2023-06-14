#!/bin/bash

#for B in "bini" "angas" "agatu"
for B in "bini"
do
    #for neighbor in 10 20 35 10000 # 10000 corresponds to finetuning on the whole training dataset 
    for neighbor in 10
    do
        #sbatch train_accent.sh $B $neighbor
        bash train_accent.sh $B $neighbor

    done
done
