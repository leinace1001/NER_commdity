import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertForMaskedLM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def position_encoding(word_ids, embedding_dim):
    B, T = word_ids.shape
    encoding = torch.zeros(B, T, embedding_dim).to(word_ids.device)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).to(word_ids.device)
    encoding[:, :, 0::2] = torch.sin(word_ids[:,:,None] * div_term)
    encoding[:, :, 1::2] = torch.cos(word_ids[:,:,None] * div_term)
    return encoding


class NERModel(nn.Module):  # Model for Name Entity Recognization
    def __init__(self, spacy_dim, cls, bert):
        super().__init__()
        self.bert = bert  # Pretrained BERT Model

        self.spacy_encoder = nn.Embedding(50, spacy_dim)  # to add spacy NER output information
        #self.transformer = SelfAttention(768 + spacy_dim, 8, 0.1)
        self.latent_dim = 256
        self.linear = nn.Linear(768, self.latent_dim)
        self.pos_emb_dim = 16
        self.transformer = nn.TransformerEncoderLayer(self.latent_dim + spacy_dim + self.pos_emb_dim, 8, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.outlayer = nn.Linear(self.latent_dim + spacy_dim + self.pos_emb_dim, cls)
        
        
    def forward(self, inputs, word_ids, spacy_lab, train = False): 
        x = self.bert(**inputs)["last_hidden_state"]
        bs, sq, d = x.shape
        
        if train:
            mask = torch.bernoulli(0.8 * torch.ones_like(spacy_lab))
            spacy_lab = spacy_lab * mask.int()
            x += torch.randn_like(x)*0.1
            x = self.dropout(x)
            
        spacy_lab = spacy_lab.view(-1,1)

        spacy_emb = self.spacy_encoder(spacy_lab)
        spacy_emb = spacy_emb.view(bs, sq, -1)
        
        word_ids_emb = position_encoding(word_ids, self.pos_emb_dim)
        x = self.linear(x)
        x = torch.cat((x, spacy_emb, word_ids_emb), dim=2)
        
        #x = self.transformer(x+word_ids_emb)
        x = self.transformer(x)
        out = self.outlayer(x)
        
        return out

    def freeze(self):  # Freeze part of pretrained BERT model
        self.bert.embeddings.requires_grad_ = False
        for i in range(10):
            self.bert.encoder.layer[i].requires_grad_ = False




