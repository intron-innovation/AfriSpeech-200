#!/bin/usr/env bash

# datasets=("./data/intron-test-public-6346-clean.csv" "./data/intron-dev-public-3231-clean.csv") 
# datasets=("./data/intron-dev-tiny-25-clean.csv" ) 
datasets=("./data/intron-dev-public-3231-clean.csv") 
audio_dir=/data/data/intron/
models_list=( 'jonatasgrosman/wav2vec2-large-xlsr-53-english' "facebook/wav2vec2-large-960h" \
    "jonatasgrosman/wav2vec2-xls-r-1b-english" "facebook/wav2vec2-large-960h-lv60-self" \
    "facebook/hubert-large-ls960-ft" "facebook/wav2vec2-large-robust-ft-swbd-300h" \
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus" "facebook/hubert-xlarge-ls960-ft" \
    "patrickvonplaten/wavlm-libri-clean-100h-large" )
whisper_models_list=("whisper_medium" "whisper_medium.en" "whisper_large" \
    "whisper_small" "whisper_base" "whisper_small.en" )
aws_models=("aws-transcribe" "aws-transcribe-medical")
gcp_azure_models=("gcp-transcribe-medical" "gcp-transcribe" "azure-transcribe")
domains=("general" "clinical" "all" ) 
# domains=("clinical") 


# python3 src/train/train.py -c src/config/config_xlsr_group_lengths.ini

# python3 src/train/train.py -c src/config/config_xlsr.ini


for dataset in ${datasets[@]}; 
do
    echo dataset: $dataset
    
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
    
    
    
#     for model in ${whisper_models_list[@]}; 
#     do
#         echo model: $model
#         python3 src/inference/whisper-inference.py --audio_dir $audio_dir --gpu 1 \
#             --model_id_or_path $model --data_csv_path $dataset --batchsize 8
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

done

python3 src/train/train.py -c src/config/config_xlsr_all_unfreeze_encoder.ini
