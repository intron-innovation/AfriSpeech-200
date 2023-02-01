import pandas as pd
from typing import List
from ast import literal_eval
data = pd.read_csv('results/ner/intron-test-public-6346-clean_with_named_entity.csv')
has_entity = data[data['has_entity'] == 1]['entities'].tolist()

def build_dataset(entities_spans_list: List, dataset_name: str):
    all_entities = []
    for entities_span in entities_spans_list:
        for entity_span in literal_eval(entities_span):
            all_entities.append('{} {}'.format(entity_span['word'].strip(), entity_span['entity'].strip())) 
        all_entities.append('')
    
    with open('ner_dataset_{}.txt'.format(dataset_name), encoding='utf-8', mode='w') as datafile:
        for entity in all_entities:
            datafile.write(entity + '\n')
    datafile.close()

build_dataset(has_entity, 'afrispeech')