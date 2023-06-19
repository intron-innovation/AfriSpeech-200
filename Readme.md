# AfriSpeech-200

#### Pan-African accented speech dataset for clinical and general domain ASR

> 100+ African accents totalling  200+ hrs of audio



### How to run this code

1. Create a virtual environment `conda create -n afrispeech python=3.9`

2. Activate the virtual environment `conda activate afrispeech`

3. Install pytorch for your operating system by following https://pytorch.org/, e.g. `pytorch==1.8.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch`

4. Install ffmpeg `sudo apt install ffmpeg`

5. Install requirements `pip3 install -r requirements.txt`

6. For Inference Run `python3 src/inference/afrispeech-inference.py --audio_dir /data/data/intron/ --model_id_or_path facebook/wav2vec2-large-960h`



#### How to run multi-task experiments

- Checkout the `wav2vec2_multitask` branch or pull the latest changes

- Update the huggingface/transformers cache directory at the top of `src/train/train.py`, `src/utils/prepare_dataset.py`, and `src/inference/afrispeech-inference.py`

- Navigate to the config directory `cd src/config/`

- Make a copy of `config_xlsr_group_lengths_multi_task.ini` and rename it to match the ablation you want to run. For example, if you want to run ASR + domain prediction, you can name it `xlsr_multi_task_asr_domain.ini`

- In the new config file, update 3 sections: 
1. Under the `experiment` section, set a unique experiment name `name` . Also set the `repo_root`, this is the location of the cloned repo. Optionally, set `dir` which is the path to your experiment directory. This is where your training artefacts will be stored, e.g. processor, vocab, checkpoints, etc. 
2. Under the `audio` section Set the `audio_path` which is the directory where you downloaded the audio files. See directions for downloading the audio files in the readme section above.
3. Under the `tasks` section, choose the tasks for your ablation. For example,  if you want to run ASR + domain prediction, set `domain=True` and set accent and vad to False.

- to start training, navigate to the repo root `cd ../..` and run `python3 src/train/train.py -c src/config/xlsr_multi_task_asr_domain.ini`

- After training, to run inference on dev set, run `python3 src/inference/afrispeech-inference.py --model_id_or_path <PATH/TO/MODEL/CHECKPOINT> --gpu 1 --batchsize 8 --audio_dir /data/data/intron/ --data_csv data/intron-test-public-6346-clean.csv`

- Upload your best model to a google drive folder and share with the team. To upload your model, create a new directory inside your experiment directory and copy the following files into this directory: best checkpoint (`pytorch_model.bin`) along with its `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`, and your experiment config file e.g. `xlsr_multi_task_asr_domain.ini`. Zip the folder using `tar czf name_of_archive_file.tar.gz name_of_directory_to_tar` and upload the tar.gz file to google drive, update the permissions, and share the link


 
#### License

&copy; 2022. This work is licensed under a CC BY-NC-SA 4.0 license.