import librosa
import numpy as np
import warnings
import traceback
import sys

warnings.filterwarnings('ignore')


class AudioConfig:
    sr = 16000
    duration = 3.0  # secs
    min_duration = 3.0  # secs
    min_array_len = sr * min_duration
    max_duration = 17 # secs
    max_array_len = sr * max_duration


def pad_zeros(x, size):
    if len(x) >= size:  # long enough
        return x
    else:  # pad blank
        return np.pad(x, (0, max(0, size - len(x))), "constant")


def load_audio_file(file_path):
    try:
        data, sr = librosa.core.load(file_path, sr=AudioConfig.sr)
        if sr != AudioConfig.sr:
            data = librosa.resample(data, sr, AudioConfig.sr)
        if len(data) > (AudioConfig.max_array_len):
            data = data[:AudioConfig.max_array_len]
        elif len(data) < (AudioConfig.min_array_len):
            data = pad_zeros(data, int(AudioConfig.min_array_len))
    except Exception as e:
        print(f"audio: {file_path} not found {str(e)}")
        print(traceback.format_exc())
        print(sys.exc_info()[2])
        data = np.random.rand(AudioConfig.sr * 3).astype(np.float32)
    return data
