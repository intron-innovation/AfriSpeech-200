from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
import numpy as np
import json

tokenizer = AutoTokenizer.from_pretrained(
    "masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0"
)
model = AutoModelForTokenClassification.from_pretrained(
    "masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0"
)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)


def extract_entities(csv, threshhold=0.8):
    df = pd.read_csv(csv)
    print(df.shape)
    
    for id in df.index:
        transcript = df.loc[id, "transcript"]
        entities = nlp(transcript)

        per_list = []
        loc_list = []
        org_list = []
        other_entity = []

        has_entity = 0

        if len(entities) != 0:
            # fix for to ensure that entities is successfully saved as json blob.
            for i in range(len(entities)):
                y = entities[i]
                for x in y.keys():
                    if type(y[x]) == np.float32:
                        y[x] = float(y[x])
                entities[i] = y
            entities_group = nlp.group_entities(entities)
            # save entities
            for el in entities_group:
                entity = el["entity_group"]
                score = el["score"]
                word = el["word"]

                if entity == "PER":
                    if score > threshhold:
                        per_list.append(word)
                        has_entity = 1

                elif entity == "LOC":
                    if score > threshhold:
                        loc_list.append(word)
                        has_entity = 1

                elif entity == "ORG":
                    if score > threshhold:
                        org_list.append(word)
                        has_entity = 1
                else:
                    other_entity.append({"entity": entity, "word": word})

        df.loc[id, "has_entity"] = has_entity
        df.loc[id, "PER"] = f"{per_list}" if len(per_list) != 0 else ""
        df.loc[id, "LOC"] = f"{loc_list}" if len(loc_list) != 0 else ""
        df.loc[id, "ORG"] = f"{org_list}" if len(org_list) != 0 else ""
        df.loc[id, "entities"] = json.dumps(entities) if len(entities) != 0 else ""

    df["has_entity"] = df["has_entity"].astype(int)
    df.to_csv(
        f"./results/ner/intron-test-public-6346-clean_with_named_entity.csv", index=None
    )
    return df


csv = "./data/intron-test-public-6346-clean.csv"
df_with_ner_tag = extract_entities(csv, threshhold=0.8)
