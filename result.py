import torch
import torch.nn.functional as F
import pandas as pd
from csv import QUOTE_NONE
from transformers import AutoTokenizer, BertForMaskedLM
import argparse

from data_tool import *
from dataset import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="2023 eBay ML Challenge Data/Listing_Titles.tsv")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
bert = BertForMaskedLM.from_pretrained("bert-base-german-cased")
bert = bert.bert
model = NERModel(64, 35, bert).to(device)
model.load_state_dict(torch.load("save/train/final.pth", map_location=device))
nlp = spacy.load("de_core_news_sm")
nlp.add_pipe("entity_ruler", before="ner").from_disk("save/spacy_model")

titles = pd.read_csv(args.data_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=QUOTE_NONE)

# Or else, if you just want to generate a subset of your data, uncommand the following line and change the index range
# titles = titles.iloc[5000:30000]

lines = []
for _, row in titles.iterrows():
    entity = process_title(row["Title"], nlp)
    title = []
    named_entity = []
    for e in entity:
        if e[2] != 'No Tag':
            title.append(e[0])
            if e[2] == None:
                named_entity.append(0)
            else:
                named_entity.append(e[2])
    
    inputs = tokenizer(title, is_split_into_words=True, return_tensors="pt")
    word_ids = []
    spacy_labels = []
    for id in inputs.word_ids():
        if id == None:
            word_ids.append(-1)
            spacy_labels.append(0)
        else:
            word_ids.append(id)
            spacy_labels.append(spacy_ent_label[named_entity[id]])
    pd_label = []
    with torch.no_grad():
        out = model(inputs.to(device), torch.tensor(word_ids).unsqueeze(0).to(device), torch.tensor(spacy_labels).unsqueeze(0).to(device))
        token_label = out.argmax(dim=2)
        
        word_label = []
        pre_id = -1
        for i,id in enumerate(word_ids):
            if id != pre_id and len(word_label)>0:
                label = torch.mode(torch.tensor(word_label)).values.item()
                pd_label.append(label)
                word_label.clear()
                word_label.append(token_label[0,i].item())
            elif id != -1:
                word_label.append(token_label[0,i].item())
            pre_id = id
    id = 0
    for e in entity:
        if e[2]!='No Tag':
            e[2] = Tags[pd_label[id]]
            id+=1

    entity_text = ""
    entity_label = 'No Tag'

    for e in entity:
        if e[2] == entity_label:
            if  entity_label != 'No Tag':
                entity_text += " "
                entity_text += e[0]
        else:
            if entity != "" and entity_label!='No Tag':
                line = "{}\t{}\t{}\n".format(row["Record Number"], entity_label, entity_text)
                lines.append(line)
            entity_label = e[2]
            if e[2] == 'No Tag':
                entity_text = ""
            else:
                entity_text = e[0]

    if entity != "" and entity_label!='No Tag':
                line = "{}\t{}\t{}\n".format(row["Record Number"], entity_label, entity_text)
                lines.append(line)

with open("save/result.csv", "w") as f:
    for line in lines:
        f.write(line)

    



