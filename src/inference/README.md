### How to run Emddings computations 

1. Set the configuration of the fine_tuned model architecture in the scr/train/model.py file.
2. Run this sample script line `python3 src/inference/run_accent_embedding.py --model_id_or_path /data/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-multi-task-3-heads-weighted-7-2-1/checkpoints/ --gpu 1 --batchsize 8 --audio_dir /data/data/intron/ --data_csv data/intron-dev-public-3231-clean.csv`