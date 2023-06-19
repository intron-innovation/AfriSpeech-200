# AfriSpeech-200

#### Pan-African accented speech dataset for clinical and general domain ASR

> 100+ African accents totalling  200+ hrs of audio


### How to Access the Data

Train and dev sets have been uploaded to an s3 bucket for public access.
Here are the steps to access the data

1. If not installed already, download and install `awscli` for your 
platform (linux/mac/windows) following the instructions [here](https
://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) 

2. Create a download folder e.g. `mkdir AfriSpeech-100`

3. Request aws credentials to access the data by sending an email
with title "AfriSpeech S3 Credentials Request" to tobi@intron.io or send me a DM on slack

4. Once you receive credentials, change into your data directory `cd AfriSpeech-100`

5. Type `aws configure` at the command line, hit enter, and fill in the following based on the credentials you receive.
    ```
    AWS Access Key ID [None]: <ACCESS-KEY-ID-FROM-CREDENTIALS-SENT>
    AWS Secret Access Key [None]: <SECRET-KEY-FROM-CREDENTIALS-SENT>
    Default region name [None]: eu-west-2
    Default output format [None]: <leave this blank>
    ```

6. Run `aws s3 cp s3://intron-open-source/AfriSpeech-100 . --recursive` to download all the audio

7. Download may take over 2hrs depending on your bandwidth. Train set: 57816, 103G; Dev set: 3227, 5.2G


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