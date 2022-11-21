import re
import os
from pathlib import Path
import librosa
import json


def cleanup(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = text.lower()
    text = text.replace(" comma,", ",") \
        .replace(" full stop.", ".") \
        .replace(" full stop", ".") \
        .replace(",.", ".") \
        .replace(",,", ",")
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    return text


def speech_file_to_array_fn(batch):
    """
    load speech array from wav file
    :param batch: dict
    :return: dict
    """
    speech_array, sampling_rate = librosa.load(batch["audio_paths"], sr=16_000)
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, sampling_rate, 16000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch


def get_s3_file(s3_file_name,
                s3_prefix="http://speech-app.s3.amazonaws.com/static/audio/uploads",
                local_prefix="s3",
                bucket_name=None, s3=None):
    """
    download file from s3 bucket
    :param s3_file_name:
    :param s3_prefix:
    :param local_prefix:
    :param bucket_name:
    :param s3:
    :return:
    """
    local_file_name = s3_file_name.replace(s3_prefix, local_prefix)
    if not os.path.isfile(local_file_name):
        Path(os.path.dirname(local_file_name)).mkdir(parents=True, exist_ok=True)
        s3_key = s3_file_name[54:]
        s3.Bucket(bucket_name).download_file(Key=s3_key, Filename=local_file_name)
    return local_file_name


def get_json_result(local_file_name):
    with open(local_file_name, 'r') as f:
        result = json.load(f)
    pred = result['results']['transcripts'][0]['transcript']
    return pred
