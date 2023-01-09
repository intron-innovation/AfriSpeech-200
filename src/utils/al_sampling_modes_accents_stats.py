import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    return dict(sorted(accent_frequencies.items(), key=lambda item: item[1], reverse=True)), accents


fig, axs = plt.subplots(al_rounds, 2, figsize=(15, 15))


def plot_frequencies(dict_accents, list_accents, round_al):
    accents_words = ''
    accents_words += " ".join(accent for accent in list_accents) + " "
    accents = list(dict_accents.keys())[:top_k_accents]
    frequencies = list(dict_accents.values())[:top_k_accents]

    accentcloud = WordCloud(width=800, height=800,
                            background_color='white', collocations=False,
                            min_font_size=10).generate(accents_words)

    axs[round_al][0].bar(accents, frequencies, label=accents,
                         color=['black', 'brown', 'salmon', 'orange', 'gold', 'red', 'green', 'blue', 'cyan',
                                'darkorange', 'lime', 'darkslateblue', 'purple', 'gray', 'pink'])
    axs[round_al][0].axes.get_xaxis().set_visible(False)
    axs[round_al][0].set_ylabel('Frequency')
    axs[round_al][0].set_title(
        'AL Round {}'.format(round_al + 1))
    axs[round_al][0].legend(loc='upper right', prop={'size': 9})

    axs[round_al][1].imshow(accentcloud)
    axs[round_al][1].axis("off")
    axs[round_al][1].set_title("AL Round {}'s Words Cloud".format(round_al + 1))


for al_round in range(al_rounds):
    filename = top_k_format.format(k, al_round, mode)
    al_round_stats = np.load(filename, allow_pickle=True)
    audio_ids, uncertainty_stats = al_round_stats[:k], al_round_stats[k:]  # should be k+3 elements in this list
    accents_dict, accents_list = get_accents(data, audio_ids)
    plot_frequencies(accents_dict, accents_list, al_round)

fig.suptitle("Top-{} Accents for '{}' Sampling".format(top_k_accents, mode.capitalize()))
fig.savefig(
    "/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/figures/AL_{}_Sampling_Accents_Stats.png",
    bbox_inches='tight', pad_inches=.3)
plt.show()