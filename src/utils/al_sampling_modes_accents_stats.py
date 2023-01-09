import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/data/intron-train-public-58000-clean.csv')
top_k_format = "/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/checkpoints/Top-{}_AL_Round_{}_Mode_'{}'.npy"
mode = 'most'
al_rounds = 3
k = 2000
top_k_accents = 15


def get_accents(dataframe, ids_list):
    results = dataframe[dataframe['audio_ids'].isin(ids_list)]
    accents = results['accent'].tolist()
    accent_frequencies = {accent: accents.count(accent) for accent in list(set(accents))}
    return dict(sorted(accent_frequencies.items(), key=lambda item: item[1], reverse=True))


fig, axs = plt.subplots(al_rounds, 1)


def plot_frequencies(dict_accents, round_al, sampling_mode):
    accents = list(dict_accents.keys())[:top_k_accents]
    frequencies = list(dict_accents.values())[:top_k_accents]
    axs[round_al].bar(accents, frequencies)
    axs[round_al].set_xticklabels(accents, rotation=45)
    axs[round_al].set_ylabel('Frequency')
    axs[round_al].set_title(
        '{} Uncertain {} Accents Distribution for AL Round {}'.format(sampling_mode.capitalize(), top_k_accents,
                                                                      round_al + 1))


for al_round in range(al_rounds):
    filename = top_k_format.format(k, al_round, mode)
    al_round_stats = np.load(filename, allow_pickle=True)
    audio_ids, uncertainty_stats = al_round_stats[:k], al_round_stats[k:]  # should be k+3 elements in this list
    accents_dict = get_accents(data, audio_ids)
    plot_frequencies(accents_dict, al_round, mode)

plt.show()
plt.savefig("/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/figures/AL_{}_Sampling_Accents_Stats.png")