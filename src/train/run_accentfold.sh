#!/bin/bash

for neighbor in 1 10 20 35

do
    src/train/train_accent.sh  $neighbor
done