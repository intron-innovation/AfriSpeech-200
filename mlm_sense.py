import torch
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertLMHeadModel

device = "cuda"
model_id = "bert-base-multilingual-uncased"



tokenizer = BertTokenizer.from_pretrained(model_id)
model = BertLMHeadModel.from_pretrained(model_id)
BERT_MASK_TOKEN = 103 #[(103, '[MASK]')]

num_samples = 4

sentence = 'Abraham is a boy'

def get_sense_score(sentence,tokenizer,model,BERT_MASK_TOKEN,num_samples):
    '''
    IDEA
    -----------------
    PP = perplexity(P) where perplexity(P) function just computes:
        (p_1*p_*p_3*...*p_N)^(-1/N) for p_i in P

    In practice you need to do the computation in log space to avoid underflow:
        e^-((log(p_1) + log(p_2) + ... + log(p_N)) / N)
    '''
    
    tensor_input = tokenizer(sentence, return_tensors='pt')['input_ids']
    batch_input = tensor_input.repeat(num_samples, 1)

    random_ids = np.random.choice(tensor_input.size(1),num_samples)
    random_ids = torch.Tensor(random_ids).long().unsqueeze(0).T


    mask = torch.zeros(batch_input.size())
    src =  torch.ones(batch_input.size(0)).unsqueeze(0).T

    mask.scatter_(1, random_ids, src)
    masked_input = batch_input.masked_fill(mask == 1, BERT_MASK_TOKEN)
    labels = batch_input.masked_fill( masked_input != BERT_MASK_TOKEN, -100)


    output = model(masked_input, labels=labels)
    loss = output['loss'].item()
    loss = loss * (-1/len(sentence)) # this is loose and not mathematically correct. It is only being used for computational efficiency.
    score = np.exp(loss)
    return score

