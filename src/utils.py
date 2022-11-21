import re
import librosa


def cleanup(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    text = text.lower()
    return text


def speech_file_to_array_fn(batch):
    """
    load speech array from wav file
    :param batch: dict
    :return: dict
    """
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, sampling_rate, 16000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch