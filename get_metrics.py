import seaborn as sns
import matplotlib.pyplot as plt 
import os, json
import ast
import pandas as pd


TASK_FOLDER = '/home/mila/c/chris.emezue/scratch/AfriSpeech-100/task4-m1/'
exp_dir_template = 'wav2vec2-large-xlsr-53-accentfold_{}_{}{}'



accent_dict = {10:'Accent_10',20:'Accent_20',35:'Accent_35',10000:'Accent_FULL'}
accent_dict_cont = {10:'Accent_10_cont',20:'Accent_20_cont',35:'Accent_35_cont',10000:'Accent_FULL_cont'}


# for normal
s= ''
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
s= ''
for neighbor in [10 ,20 ,35]:
    s = s+f'{accent_dict_cont[neighbor]}'
    for B in ["bini" ,"angas", "agatu"]:

        exp_dir_name = exp_dir_template.format(B,neighbor,'_cont_6')
        exp_dir = os.path.join(TASK_FOLDER,exp_dir_name)

        metrics_path = os.path.join(exp_dir,'metrics-test.json')
        with open(metrics_path,'r') as f:
            metrics = json.load(f)
        wer = metrics['eval_wer']
        s = s+'&'+f'{wer:.3f}'
    s = s + '\\ \n'
breakpoint()


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