import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data = pd.read_csv('/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/data/intron-train-public-58000-clean.csv')
top_k_format = "/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/checkpoints/Top-{}_AL_Round_{}_Mode_'{}'.npy"
wers_file = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/data/Top-{}_AL_Round_{}_Mode_{}_WERS.csv'
mode = 'most'
al_rounds = 3
k = 2000
top_k_accents = 15


def get_accents(dataframe, ids_list):
    results = dataframe[dataframe['audio_ids'].isin(ids_list)]
    accents = results['accent'].tolist()
    accent_frequencies = {accent: accents.count(accent) for accent in list(set(accents))}
    accent_frequencies_sorted = dict(sorted(accent_frequencies.items(), key=lambda item: item[1], reverse=True))
    ids_audios_most_uncertain = results[results['accent'].isin(list(accent_frequencies_sorted.keys())[:top_k_accents])]
    ids_audios_least_uncertain = results[results['accent'].isin(list(accent_frequencies_sorted.keys())[top_k_accents:])]

    # print(ids_audios_most_uncertain)
    return accent_frequencies_sorted, accents, ids_audios_most_uncertain['audio_ids'].tolist(), \
        ids_audios_least_uncertain['audio_ids'].tolist()


fig, axs = plt.subplots(al_rounds, 3, figsize=(15, 15))


def fill_uncertainty(audio, list_of_audios, wers_dict):
    if audio in list_of_audios:
        return wers_dict[audio]
    else:
        return None


def plot_frequencies(dict_accents, list_accents, round_al, list_audios_high_uncertain, list_audios_low_uncertain):
    wer_file = wers_file.format(k, round_al, mode)
    data_wer = pd.read_csv(wer_file)
    audios_accents_most_uncertainty = data_wer[data_wer['audios_ids'].isin(list_audios_high_uncertain)]
    audios_accents_most_uncertainty = {audio_id: wer for audio_id, wer in
                                       zip(audios_accents_most_uncertainty['audios_ids'],
                                           audios_accents_most_uncertainty['uncertainty_wer'])}

    audios_accents_least_uncertainty = data_wer[data_wer['audios_ids'].isin(list_audios_low_uncertain)]
    audios_accents_least_uncertainty = {audio_id: wer for audio_id, wer in
                                        zip(audios_accents_least_uncertainty['audios_ids'],
                                            audios_accents_least_uncertainty['uncertainty_wer'])}

    data_wer['top-{}-uncertain-accents'.format(top_k_accents)] = data_wer['audios_ids'].apply(
        lambda x: fill_uncertainty(x, list_audios_high_uncertain, audios_accents_most_uncertainty))

    data_wer['least-uncertain-accents'] = data_wer['audios_ids'].apply(
        lambda x: fill_uncertainty(x, list_audios_low_uncertain, audios_accents_least_uncertainty))

    data_wer.rename(columns={'uncertainty_wer': 'all-accents'}, inplace=True)
    data_wer[['top-{}-uncertain-accents'.format(top_k_accents), 'least-uncertain-accents',
              'all-accents']].plot.kde(ax=axs[round_al][2])
    axs[round_al][2].legend(loc='upper right', prop={'size': 8})
    axs[round_al][2].set_title(
        "AL Round {} African Accents' WER Distribution".format(round_al + 1))
    axs[round_al][2].axes.get_xaxis().set_visible(False)

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
        'AL Round {} Top-{}-Most Uncertain African Accents'.format(round_al + 1, top_k_accents))
    axs[round_al][0].legend(loc='upper right', prop={'size': 9})

    axs[round_al][1].imshow(accentcloud)
    axs[round_al][1].axis("off")
    axs[round_al][1].set_title("AL Round {}'s African Accents' Cloud".format(round_al + 1))


for al_round in range(al_rounds):
    filename = top_k_format.format(k, al_round, mode)
    al_round_stats = np.load(filename, allow_pickle=True)
    audio_ids, uncertainty_stats = al_round_stats[:k], al_round_stats[k:]  # should be k+3 elements in this list
    accents_dict, accents_list, most_uncertain_audios, least_uncertain_audios = get_accents(data, audio_ids)
    plot_frequencies(accents_dict, accents_list, al_round, most_uncertain_audios, least_uncertain_audios)

fig.savefig(
    "/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/figures/AL_{}_Sampling_Accents_Stats.png",
    bbox_inches='tight', pad_inches=.3)
plt.show()

# get the intersection of the top-15 for the three runs
al_0, al_1, al_2 = rounds_difference_stats
common_accents = set(list(al_0.keys())[:top_k_accents]).intersection(
    set(list(al_1.keys())[:top_k_accents])).intersection(set(list(al_2.keys())[:top_k_accents]))
common_accents_list = list(common_accents)
frequencies_round_1 = [al_0[accent] for accent in common_accents_list]
frequencies_round_2 = [al_1[accent] for accent in common_accents_list]
frequencies_round_3 = [al_2[accent] for accent in common_accents_list]

df = pd.DataFrame(
    {'AL Round 1': frequencies_round_1, 'Al Round 2': frequencies_round_2, 'Al Round 3': frequencies_round_3},
    index=common_accents_list)

fig_bars, axs_bars = plt.subplots(1, 1, figsize=(15, 15))
df.plot.bar(rot=0, ax=axs_bars)
axs_bars.set_ylabel('Frequency')
fig_bars.suptitle(
    'Distribution of Most Uncertain Accents Appearing Across all 3 AL rounds (from the top-{} samples)'.format(k))
fig_bars.savefig('/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/figures/most_common_accents_distributions.png', bbox_inches='tight', pad_inches=.3)
plt.show()

all_common_accents = set(list(al_0.keys())).intersection(
    set(list(al_1.keys()))).intersection(set(list(al_2.keys())))

all_common_accents_list = list(all_common_accents)
frequencies_difference_round_0_round_1 = [al_0[accent] - al_1[accent] for accent in all_common_accents_list]
frequencies_difference_round_1_round_2 = [al_1[accent] - al_2[accent] for accent in all_common_accents_list]
difference_one = {accent: difference for accent, difference in
                  zip(all_common_accents_list, frequencies_difference_round_0_round_1)}

difference_two = {accent: difference for accent, difference in
                  zip(all_common_accents_list, frequencies_difference_round_1_round_2)}

most_variance_accents_one = dict(sorted(difference_one.items(), key=lambda item: item[1], reverse=True))
most_variance_accents_two = dict(sorted(difference_two.items(), key=lambda item: item[1], reverse=True))
all_most_variant_accents = set(list(most_variance_accents_one.keys())).intersection(
    set(list(most_variance_accents_two.keys())))
all_most_variant_accents_list = list(all_most_variant_accents)

most_variance_one = [most_variance_accents_one[accent] for accent in all_most_variant_accents_list]
most_variance_two = [most_variance_accents_two[accent] for accent in all_most_variant_accents_list]

df_variance = pd.DataFrame(
    {'Round 1 $\\rightarrow$ Round 2': most_variance_one, 'Round 2 $\\rightarrow$ Round 3': most_variance_two},
    index=all_most_variant_accents_list)

fig_bars_variance, axs_bars_variance = plt.subplots(1, 1, figsize=(20, 20))
df_variance.plot.bar(rot=90, ax=axs_bars_variance)
axs_bars_variance.set_ylabel('Frequency')
fig_bars_variance.suptitle(
    'Variational Distribution of Accents Frequencies Across all AL rounds (from the top-{} samples)'.format(k))
fig_bars_variance.savefig('/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/figures/most_varying_accents.png', bbox_inches='tight', pad_inches=.3)
plt.show()