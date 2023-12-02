import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import spacy
from torch.utils.data import Dataset
from csv import QUOTE_NONE
from transformers import BatchEncoding

from data_tool import *


class LabeledTitle(Dataset): # Tagged_Titles
    def __init__(self, file) -> None:
        super().__init__()
        raw_data = pd.read_csv(file, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=QUOTE_NONE)
        nlp = spacy.load("de_core_news_sm")
        nlp.add_pipe("entity_ruler", before="ner").from_disk("save/spacy_model")
        data = process_data(raw_data, nlp)
        inputs, spacy_label, word_ids, gt_label = tonkenize_aline(data)
        self.inputs = inputs
        self.spacy_label = spacy_label
        self.word_ids = word_ids
        self.gt_label = gt_label
        

    def __getitem__(self, index):
        return self.inputs[index], torch.LongTensor(self.spacy_label[index]), torch.tensor(self.word_ids[index]), torch.tensor(self.gt_label[index]), torch.tensor([0])
    
    def __len__(self):
        return len(self.inputs)
    
    
class UnlabeledTitle(Dataset): # Listing_Titles
    def __init__(self, raw_data) -> None:
        super().__init__()
        self.title = []
        for t in raw_data:
            entity = process_title(t)
            title = []
            for e in entity:
                if e[2] != "No Tag":
                    title.append(e[0])

            self.title.append(title)

    def __getitem__(self, index):
        return self.title[index]
    
    def __len__(self):
        return len(self.title)


class PseudoLabelTitle(Dataset):
    def __init__(self, raw_data, tokenizer, model, device, batch_size=32) -> None:
        super().__init__()
        titles = []
        named_entities = []
        nlp = spacy.load("de_core_news_sm")
        nlp.add_pipe("entity_ruler", before="ner").from_disk("save/spacy_model")
        for t in raw_data:
            entity = process_title(t,nlp)
            title = []
            named_entity = []
            for e in entity:
                if e[2] != 'No Tag':
                    title.append(e[0])
                    if e[2] == None:
                        named_entity.append(0)
                    else:
                        named_entity.append(e[2])
            titles.append(title)
            named_entities.append(named_entity)
        self.inputs = []
        
        self.spacy_label = []
        self.word_ids = []
        self.pd_label = []
        index = 0
        while index<len(raw_data):
            batch = min(batch_size, len(raw_data)-index)
            inputs = tokenizer(titles[index:index+batch], is_split_into_words=True, truncation=True, padding="max_length", max_length=80, return_tensors="pt")
            spacy_labels = []
            word_ids = []
            for i in range(batch):
                spacy_label = []
                word_id = []
                input = {}
                input['input_ids'] = inputs['input_ids'][i]
                input['token_type_ids'] = inputs['token_type_ids'][i]
                input['attention_mask']= inputs['attention_mask'][i]
                for id in inputs.word_ids(i):
                    if id == None:
                        word_id.append(-1)
                        spacy_label.append(0)
                    else:
                        word_id.append(id)
                        spacy_label.append(spacy_ent_label[named_entities[index][id]])
                index += 1
                self.inputs.append(BatchEncoding(input))
                spacy_labels.append(spacy_label)
                word_ids.append(word_id)
        
            self.spacy_label.extend(spacy_labels)
            self.word_ids.extend(word_ids)
            with torch.no_grad():
                
                pd = model(inputs.to(device), torch.tensor(word_ids).to(device), torch.LongTensor(spacy_labels).to(device))
                pd = F.softmax(pd, dim=2)
                
            for i in range(batch):
                pd_label = []
                tmp = []
                pre_id = None
                for j, id in enumerate(inputs.word_ids(i)):
                    if id != pre_id and len(tmp)>0:
                        label = torch.mode(torch.tensor(tmp)).values.item()
                        
                        for _ in tmp:
                            pd_label.append(label)
                        tmp.clear()
                    pre_id = id
                    if id == None:
                        pd_label.append(-100)
                    else:
                        value = pd[i,j,:].max()
                        if value.item()>0.7:
                            tmp.append(pd[i,j,:].argmax().item())
                        else:
                            tmp.append(-100)
                self.pd_label.append(pd_label)
                
        
    def __getitem__(self, index):
        return self.inputs[index], torch.LongTensor(self.spacy_label[index]), torch.tensor(self.word_ids[index]), torch.tensor(self.pd_label[index]), torch.tensor([1])
    
    def __len__(self):
        return len(self.inputs)
        




                    
            
            


        



