#!/bin/bash

#for B in "bini" "angas" "agatu"
for B in "bini"
do
    #for neighbor in 1 10 20 35
    for neighbor in 10
    do
        #sbatch train_accent.sh $B $neighbor
        bash train_accent.sh $B $neighbor

    done
done
