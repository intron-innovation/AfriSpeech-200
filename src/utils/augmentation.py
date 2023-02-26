from audiomentations import Compose, AddGaussianNoise
import librosa
import numpy as np
import argparse
import pandas as pd
import random
from scipy.io import wavfile

"""
what does this script do?
augments two kinds of audio files and csv files.
1. adds gaussian noise to the audio
2. non speech data
how to use: python3 augmentation.py --csv /AfricanNLP/intron-dev-public-3232.csv --output_csv augmented_dev.csv --output_audio augmented_audio
"""

def augment_sample(min_amp=0.001, max_amp=0.015, prob=0.5, duration=5, sample=None):

    augment = Compose([
        AddGaussianNoise(min_amplitude=min_amp, max_amplitude=max_amp, p=prob),
    ])

    if not sample:
        samples = np.random.uniform(low=-0.2, high=0.2, size=(duration*16000,)).astype(np.float32)
        # Augment/transform/perturb the audio data
        augmented_samples = augment(samples=samples, sample_rate=16000)
    else:
        y, sr = librosa.load(sample, sr=16000)
        # Augment/transform/perturb the audio data
        augmented_samples = augment(samples=y, sample_rate=16000)
    return augmented_samples



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="csv with transcript")
    parser.add_argument("--output_csv", type=str, help="output csv file")
    parser.add_argument("--output_audio", type=str, help="output audio folder")
    args = parser.parse_args()
    original_df = pd.read_csv(args.csv)
    random.seed(10)
    # original_df = original_df[:20]
    dout = original_df
    samples = []
    transcripts = ["<UNK>"]*len(original_df)
    accents = ["<UNK>"]*len(original_df)
    domain = ["<UNK>"]*len(original_df)
    durations = []
    for i in range(len(dout)):
        # dur = random.uniform(3, 17)
        temp_sample = dout.loc[i, "audio_paths"]
        inp_sample = temp_sample.replace("/AfriSpeech-100/", "/Users/adiyadaval/AfricanNLP/audio/")
        data = augment_sample(sample=inp_sample)
        wavfile.write(args.output_audio+"/"+inp_sample.split("/")[-1].split(".")[0] + "_augmented.wav", 16000, data)
        samples.append(args.output_audio+"/"+inp_sample.split("/")[-1].split(".")[0] + "_augmented.wav")
    dout["audio_paths"] = samples
    dout.to_csv("augmented_"+args.output_csv, index=False)
    samples = []
    for i in range(len(original_df)):
        dur = random.randint(3, 17)
        durations.append(dur)
        data = augment_sample(duration=dur)
        temp_sample = original_df.loc[i, "audio_paths"]
        inp_sample = temp_sample.replace("/AfriSpeech-100/", "/Users/adiyadaval/AfricanNLP/audio/")
        wavfile.write(args.output_audio+"/"+inp_sample.split("/")[-1].split(".")[0] + "_ns.wav", 16000, data)
        samples.append(args.output_audio+"/"+inp_sample.split("/")[-1].split(".")[0] + "_ns.wav")
    ns_df = pd.DataFrame(
    {'duration': durations,
    'audio_paths': samples,
    'domain': domain,
    'accent': accents,
    'transcripts': transcripts,
    })
    ns_df.to_csv("ns_"+args.output_csv, index=False)