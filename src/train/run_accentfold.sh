#!/bin/bash

for neighbor in 1 10 20 35

do
    bash src/train/train_accent.sh  $neighbor
done