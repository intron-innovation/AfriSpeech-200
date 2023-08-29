import seaborn as sns
import matplotlib.pyplot as plt 
import os, json
import ast
import pandas as pd
from plot_table import plot_table,get_color_shade
import scipy.stats as stats


TASK_FOLDER = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4/'
exp_dir_template = 'wav2vec2-large-xlsr-53-accentfold_{}_{}{}'



accent_dict = {10:'Accent\_10',20:'Accent\_20',35:'Accent\_35',10000:'Accent\_FULL'}
accent_dict_cont = {10:'Accent\_10\_cont',20:'Accent\_20\_cont',35:'Accent\_35\_cont',10000:'Accent\_FULL_\cont'}
accent_dict_random = {10:'Accent\_10\_random',20:'Accent\_20\_random',35:'Accent\_35\_random',10000:'Accent\_FULL\_random'}


# for normal
#s= ''
# for neighbor in [10 ,20 ,35,10000]:
#     s = s+f'{accent_dict[neighbor]}'
#     for B in ["bini" ,"angas", "agatu"]:

#         exp_dir_name = exp_dir_template.format(B,neighbor,'_')
#         exp_dir = os.path.join(TASK_FOLDER,exp_dir_name)

#         metrics_path = os.path.join(exp_dir,'metrics-test.json')
#         with open(metrics_path,'r') as f:
#             metrics = json.load(f)
#         wer = metrics['eval_wer']
#         s = s+'&'+f'{wer:.3f}'
#     s = s + '\\ \n'
# breakpoint()

# # for CONT
# s= ''
# for neighbor in [10 ,20 ,35]:
#     s = s+f'{accent_dict_cont[neighbor]}'
#     for B in ["bini" ,"angas", "agatu"]:

#         exp_dir_name = exp_dir_template.format(B,neighbor,'_cont_6')
#         exp_dir = os.path.join(TASK_FOLDER,exp_dir_name)

#         metrics_path = os.path.join(exp_dir,'metrics-test.json')
#         with open(metrics_path,'r') as f:
#             metrics = json.load(f)
#         wer = metrics['eval_wer']
#         s = s+'&'+f'{wer:.3f}'
#     s = s + '\\ \n'
# breakpoint()


# for RANDOM
# s= ''
# for neighbor in [10 ,20 ,35]:
#     s = s+f'{accent_dict_random[neighbor]}'
#     for B in ["bini" ,"angas", "agatu"]:

#         exp_dir_name = exp_dir_template.format(B,neighbor,'_random')
#         exp_dir = os.path.join(TASK_FOLDER,exp_dir_name)

#         try:
#             metrics_path = os.path.join(exp_dir,'metrics-test.json')
#             with open(metrics_path,'r') as f:
#                 metrics = json.load(f)
#             wer = metrics['eval_wer']
#         except FileNotFoundError:
#             wer = 100.000
#         s = s+'&'+f'{wer:.3f}'
#     s = s + '\\ \n'
# breakpoint()


#breakpoint()

def get_details_of_accent_subset(plot=False):
    # #train samples
    # # accents
    # list the accents
    wers=[]
    ts=[]
    acc=[]
    for B in ["bini" ,"angas", "agatu"]:

        for neighbor in [10 ,20 ,35]:
            acc.append(B)    
            exp_dir_name = exp_dir_template.format(B,neighbor,'_')
            exp_dir = os.path.join(TASK_FOLDER,exp_dir_name)

            metrics_path = os.path.join(exp_dir,'metrics-test.json')
            with open(metrics_path,'r') as f:
                metrics = json.load(f)
            print('='*20 + f'{B}' + '='*20)
            accents_details = ast.literal_eval(metrics["train_accent_subset"])
            print(f'# WER: {metrics["eval_wer"]:.3f}')
            wers.append(metrics["eval_wer"])
            print(f'# Train samples: {metrics["size_train_dataset"]}')
            ts.append(metrics["size_train_dataset"])
            print(f'# Accents in subset: {len(accents_details)}')
            print(f'# The accents in subset: {accents_details}')
    if plot:
        breakpoint()

        df = pd.DataFrame({'Test WER':wers,'Accent':acc,'Train Size':ts})
        fig,ax = plt.subplots()
        ax =sns.relplot(
            data=df, x="Train Size", y="Test WER",
            col="Accent", kind="line"
        )
        #ax.set_titles('Scatterplot of test WER against training dataset size')
        plt.legend()
        ax.savefig('scatterplot_line_testwer_ts.png')

        breakpoint()



#get_details_of_accent_subset(True)



def get_completed_runs(folder):
    run_folders = [os.path.join(folder,f) for f in os.listdir(folder)]
    metric_paths = [os.path.join(f,'metrics-test.json') for f in run_folders]
    exist_metric_paths = [1 if os.path.exists(f) else 0 for f in metric_paths]
    print(f'metrics json files completed: {sum(exist_metric_paths)} \ {len(run_folders)}')
    #for p, m in zip(metric_paths,exist_metric_paths):
    #    if m==0:
    #        print(p)


#TASK_FOLDER_RANDOM = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-new-idea/'

TASK_FOLDER_RANDOM = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-geo-prox/'

test_only_accents = ['obolo', 'jukun', 'bini', 'etche', 'bajju', 'idah', 'ikulu', 'ukwuani', 'estako', 'ekene', 'okirika', 'ishan', 'eket', 'ibani', 'eleme', 'yoruba, hausa', 'eggon', 'ebiobo', 'mada', 'nyandang', 'ijaw(nembe)', 'agatu', 'gbagyi', 'urobo', 'yala mbembe', 'ekpeye', 'gerawa', 'bassa', 'afo', 'mwaghavul', 'kubi', 'igbo and yoruba', 'bagi', 'jaba', 'khana', 'angas', 'brass', 'delta', 'oklo', 'kalabari', 'igarra']

get_completed_runs(TASK_FOLDER_RANDOM)
breakpoint()

def read_json(file_path):
    with open(file_path,'r') as f:
        return json.load(f)






def get_all_test_accent_stats():
    folder = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-new-idea/'
    # we want to get random, af, accent

    exp_folders = [(os.path.join(folder,f),'random') if 'random' in f else (os.path.join(folder,f),'AccentFold') for f in os.listdir(folder)]
    metric_paths = [(os.path.join(f[0],'metrics-test.json'),f[1]) for f in exp_folders if os.path.exists(os.path.join(f[0],'metrics-test.json'))]
    metrics = [(read_json(m[0]),m[1]) for m in metric_paths]
    accents = [m[0]['b'] for m in metrics]
    wer = [m[0]['eval_wer'] for m in metrics]
    eval_samples = [m[0]['eval_samples'] for m in metrics]
    size_train_dataset = [m[0]['size_train_dataset'] for m in metrics]
    eval_loss = [m[0]['eval_loss'] for m in metrics]
    method = [m[1] for m in metrics]

    df = pd.DataFrame({'Method':method,'Accent':accents,'Test WER':wer,'# Test samples':eval_samples,'# Train samples':size_train_dataset,'Test loss':eval_loss})
    # Only select the 41 accents in test
    # test_adjust = ['gerawa','igbo and yoruba','obolo','ukwuahi','urobo','ikulu']

    
    df_test_only = df[df['Accent'].isin(test_only_accents)]
    # df[df['Accent'].isin(test_adjust)]
    return df_test_only



def get_all_test_accent_stats_geo():
    folder = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-geo-prox/'

    exp_folders = [(os.path.join(folder,f),'geo') for f in os.listdir(folder)]
    metric_paths = [(os.path.join(f[0],'metrics-test.json'),f[1]) for f in exp_folders if os.path.exists(os.path.join(f[0],'metrics-test.json'))]
    metrics = [(read_json(m[0]),m[1]) for m in metric_paths]
    accents = [m[0]['b'] for m in metrics]
    wer = [m[0]['eval_wer'] for m in metrics]
    eval_samples = [m[0]['eval_samples'] for m in metrics]
    size_train_dataset = [m[0]['size_train_dataset'] for m in metrics]
    eval_loss = [m[0]['eval_loss'] for m in metrics]
    method = [m[1] for m in metrics]

    df = pd.DataFrame({'Method':method,'Accent':accents,'Test WER':wer,'# Test samples':eval_samples,'# Train samples':size_train_dataset,'Test loss':eval_loss})
    # Only select the 41 accents in test
    # test_adjust = ['gerawa','igbo and yoruba','obolo','ukwuahi','urobo','ikulu']

    
    df_test_only = df[df['Accent'].isin(test_only_accents)]
    # df[df['Accent'].isin(test_adjust)]
    return df_test_only

#df = get_all_test_accent_stats()
#breakpoint()
df = get_all_test_accent_stats_geo()
print('Mean ', df.groupby(['Method']).mean())
breakpoint()

# perform one sample t-test
t_statistic, p_value = stats.ttest_1samp(a=df['Test WER'].values.tolist(), popmean=30)

# df.groupby(['Method']).mean()
# fig, ax = plt.subplots()
# ax = sns.lineplot(data=df, x="Accent", y="Test WER",style="Method",hue='Method',markers=True)

# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
# fig.set_figwidth(13.4)
# fig.tight_layout()
# fig.savefig('lineplot_test_wer_41.png')

# fig, ax = plt.subplots()
# ax = sns.barplot(data=df, y="# Train samples", x="Accent", hue="Method")

# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
# fig.set_figwidth(13.4)
# fig.tight_layout()
# fig.savefig('train_samples_barplot_test_41.png')
# breakpoint()


def compare_accent_subsets():
    folder = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-new-idea/'
    geo_folder = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-geo-prox/'

    exp_folders = [(os.path.join(folder,f),'random') if 'random' in f else (os.path.join(folder,f),'AccentFold') for f in os.listdir(folder)]
    af_exp_folders = [m for m in exp_folders if m[1]=='AccentFold']

    geo_exp_folders = [(os.path.join(geo_folder,f),'geo') for f in os.listdir(geo_folder)]

    af_metric_paths = [(os.path.join(f[0],'metrics-test.json'),f[1]) for f in af_exp_folders if os.path.exists(os.path.join(f[0],'metrics-test.json'))]
    geo_metric_paths = [(os.path.join(f[0],'metrics-test.json'),f[1]) for f in geo_exp_folders if os.path.exists(os.path.join(f[0],'metrics-test.json'))]

    af_metrics = [(read_json(m[0]),m[1]) for m in af_metric_paths]
    af_metrics = [m[0] for m in af_metrics]

    geo_metrics = [(read_json(m[0]),m[1]) for m in geo_metric_paths]
    geo_metrics = [m[0] for m in geo_metrics]

    # create lang and counts. 
    accents = []
    counts = []
    count=0
    diff=[]
    for accent in test_only_accents:
        
        af_specific_metric = [a for a in af_metrics if a['b']==accent]
        geo_specific_metric = [a for a in geo_metrics if a['b']==accent]

        if af_specific_metric==[] or geo_specific_metric==[] :
            continue
        assert len(af_specific_metric)==1 and len(geo_specific_metric)==1
        af_accent_specific_metric = af_specific_metric[0]
        geo_accent_specific_metric = geo_specific_metric[0]

        af_accents_used = ast.literal_eval(af_accent_specific_metric['train_accent_subset'])
        geo_accents_used = ast.literal_eval(geo_accent_specific_metric['train_accent_subset'])
        different_af = len(set(geo_accents_used))-len(set(af_accents_used).intersection(set(geo_accents_used)))
        diff.append(different_af)
        print(f'Intersection for accent {accent}: ',different_af)
        count+=1
        #assert len(accents_used) == 20
    print('Count: ',count)
    fig,ax = plt.subplots()
    ax.hist(diff)
    ax.set_title('# accents from AccentFold different from GeoProx')
    ax.set_xlabel('# accents')
    ax.set_ylabel('Count')
    fig.savefig('plots/hist_diff_accents_fold_geo.png')
    breakpoint()


def get_table_chosen_accents(type_filter='AccentFold'):
    folder = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-new-idea/'

    exp_folders = [(os.path.join(folder,f),'random') if 'random' in f else (os.path.join(folder,f),'AccentFold') for f in os.listdir(folder)]
    metric_paths = [(os.path.join(f[0],'metrics-test.json'),f[1]) for f in exp_folders if os.path.exists(os.path.join(f[0],'metrics-test.json'))]
    metrics = [(read_json(m[0]),m[1]) for m in metric_paths]
    metrics = [m[0] for m in metrics if m[1]==type_filter]



    # Get dict of accents and their # clips
    df_d = pd.read_csv('afrispeech_stats.csv')
    df_d_accents = df_d['Accent'].values.tolist()
    df_d_count = df_d['Clips'].values.tolist()
    accent_dict = {k:v for k,v in zip(df_d_accents,df_d_count)}


    # create lang and counts. 
    accents = []
    counts = []
    for accent in test_only_accents:
        accents.append(f'{accent}_MAINK')
        counts.append(0)
        specific_metric = [a for a in metrics if a['b']==accent]
        if specific_metric==[]:
            continue
        assert len(specific_metric)==1
        accent_specific_metric = specific_metric[0]
        accents_used = ast.literal_eval(accent_specific_metric['train_accent_subset'])
        assert len(accents_used) == 20
        accents.extend(accents_used)
        counts.extend([accent_dict[a] for a in accents_used])
    assert len(accents)==len(counts)

    fig,the_table,colors_ = plot_table(accents,counts,21)
    the_table.set_fontsize(30)
    the_table.figure.set_figwidth(12.8)
    the_table.figure.set_figheight(9.6)

    #the_table.scale(1,5)
    #plt.title('Heatmap ofLanfrica Records per language')
    the_table.figure.tight_layout()

    the_table.figure.savefig('plots/table_accent_chosen_heatmap.svg')
    #plt.show()

    breakpoint()

compare_accent_subsets()
#get_table_chosen_accents()