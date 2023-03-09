#!/bin/usr/env bash

datasets=("./data/intron-test-public-6346-clean.csv" "./data/intron-dev-public-3231-clean.csv") 
# datasets=("./data/intron-dev-tiny-25-clean.csv" ) 
# datasets=("./data/intron-dev-public-3231-clean.csv") 
# datasets=("librispeech") 
audio_dir=/data/data/intron/
models_list=( "facebook/wav2vec2-large-robust-ft-libri-960h" \
    'jonatasgrosman/wav2vec2-large-xlsr-53-english' "facebook/wav2vec2-large-960h" \
    "jonatasgrosman/wav2vec2-xls-r-1b-english" "facebook/wav2vec2-large-960h-lv60-self" \
    "facebook/hubert-large-ls960-ft" "facebook/wav2vec2-large-robust-ft-swbd-300h" \
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus" "facebook/hubert-xlarge-ls960-ft" \
    "patrickvonplaten/wavlm-libri-clean-100h-large" )
whisper_models_list=("whisper_medium" "whisper_medium.en" "whisper_large" \
    "whisper_small" "whisper_base" "whisper_small.en" )
aws_models=("aws-transcribe" "aws-transcribe-medical")
gcp_azure_models=("azure-transcribe") # "gcp-transcribe-medical" "gcp-transcribe"  
domains=("general" "clinical" "all" ) 
# models_list=( "facebook/wav2vec2-large-robust-ft-libri-960h")
# domains=("clinical") 

# for dataset in ${datasets[@]}; 
# do
#     echo dataset: $dataset

#     for model in ${models_list[@]}; 
#     do
#         echo model: $model
#         python3 src/inference/whisper-inference.py --audio_dir $audio_dir --gpu 1 \
#             --model_id_or_path $model --data_csv_path $dataset --batchsize 8
#     done
    
    
    
#     for model in ${whisper_models_list[@]}; 
#     do
#         echo model: $model
#         python3 src/inference/whisper-inference.py --audio_dir $audio_dir --gpu 1 \
#             --model_id_or_path $model --data_csv_path $dataset --batchsize 8
#     done
        
    
#     for domain in ${domains[@]}; 
#     do
#         echo model: $domain
#         python3 src/inference/whisper-inference.py --audio_dir $audio_dir --gpu 1 \
#             --model_id_or_path ./src/experiments/whisper_$domain/ \
#             --data_csv_path $dataset --batchsize 8
#     done


#     for domain in ${domains[@]}; 
#     do
#         echo model: $domain
#         python3 src/inference/whisper-inference.py --audio_dir $audio_dir --gpu 1 \
#             --model_id_or_path ./src/experiments/wav2vec2-large-xlsr-53-$domain/checkpoints/ \
#             --data_csv_path $dataset --batchsize 8
#     done
    

    
#     for model in ${aws_models[@]}; 
#     do
#         echo model: $model
#         python3 bin/aws_transcribe.py --model_id_or_path $model \
#          --audio_dir $audio_dir --data_csv_path $dataset
#     done
    
    
    
#     for model in ${gcp_azure_models[@]}; 
#     do
#         echo model: $model
#         python3 bin/gcp_speech_api.py --model_id_or_path $model \
#             --audio_dir $audio_dir --data_csv_path $dataset
#     done

# done


# python3 src/train/train.py -c src/config/config_xlsr_group_lengths.ini

# python3 src/train/train.py -c src/config/config_xlsr.ini

# python3 src/train/train.py -c src/config/config_xlsr_all_unfreeze_encoder.ini

# python3 src/train/train.py -c src/config/config_xlsr_group_lengths_multi_task-prepend.ini
# python3 src/train/train.py -c src/config/config_xlsr_group_lengths_multi_task-append.ini


model_list=(
#     "wav2vec2-large-xlsr-53-generative-multitask-asr-domain-append" \
#     "wav2vec2-large-xlsr-53-generative-multitask-asr-domain-prepend" \
#     "wav2vec2-large-xlsr-53-generative_single_task_baseline" \
#     "wav2vec2-large-xlsr-53-generative-multitask-asr-accent-append" \
#     "wav2vec2-large-xlsr-53-generative-multitask-asr-accent-prepend" \
#     "wav2vec2-large-xlsr-53-generative-multitask-asr-accent-prepend-repeat" \
#     "wav2vec2-large-xlsr-53-discriminative-asr-accent-weighted-9-1" \
#     "wav2vec2-large-xlsr-53-discriminative-single-task-baseline" \
    "wav2vec2-large-xlsr-53-multi-task-3-heads-mean" \
    "wav2vec2-large-xlsr-53-multi-task-3-heads-weighted-8-1-1" \
    "wav2vec2-large-xlsr-53-multi-task-3-heads-weighted-6-2-2" \
    "wav2vec2-large-xlsr-53-discriminative-asr-accent-5-5"
)
# test_dataset=./data/intron-test-public-6346-clean.csv
test_dataset=./data/intron-dev-public-3231-clean.csv
audio_dir=/data/data/intron/

for dataset in ${datasets[@]}; 
    do
    for model in ${model_list[@]}; 
    do
        echo $model
        python3 src/inference/afrispeech-inference.py --audio_dir $audio_dir --gpu 1 \
            --model_id_or_path ./src/experiments/$model/checkpoints/ \
            --data_csv_path $dataset --batchsize 8
    done
done

# Generative vs Multiple losses
# rerun baseline
# add vad augmented samples
# append
# prepend
# ablations
# remove vad
# remove accent prediction
# remove domain prediction

# Loss weights 
# equal
# one high vs rest s
# [0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9]
# higher to ASR, rest equal