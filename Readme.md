# AccentFold | Task 4


- [Google Doc](https://docs.google.com/document/d/1MzHIB1PCe8QVrjdA-2_DBkYld-4UNhtbPVO_Op9FH0Y/edit#)


## How the Code works
The code is set up to use a given accent `B` and the number of neighbours `K` to  calculate the distance between all the accents and `B`. This returns a list (`accent_subset`) of K accents that has the closest distance including B at index 0. 


`accent_subset` is then used to filter the train_dataset while `B` is used to filter the test_dataset. The filtered train_dataset is used to finetune the model and evaluate on the filter test_dataset . Checkpoints are saved at every epoch to the dir `wav2vec2-large-xlsr-53-accentfold__{B}_{K}`. 
To finetune the model on the entire training dataset we will set K=1. </br>

To use the codebase, run `bash run_accentfold.sh`. This will call  `train_accent.sh` for multiple Ks and Bs. `train_accent.sh` contains the cli commands to finetune the models with a config file.


## How we Select the `B`
We perform evaluation with already finetuned model on the train set for all the accent in the test set and pick the accent with the highest WER with the highest number of sample that is not in the train set.

