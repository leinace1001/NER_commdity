import spacy
from spacy.lang.de import German
import pandas as pd
import csv
import os
import argparse
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

fixed_ent = ["Abteilung", "Aktivität", "Akzente", "Anlass", "Besonderheiten", 
             "Dämpfungsgrad","Farbe", "Futtermaterial", "Gewebeart", "Innensohlenmaterial",
             "Jahreszeit", "Laufsohlenmaterial", "Muster", "Obermaterial", "Produktart", 
             "Schuhschaft-Typ", "Schuhweite", "Stil", "Stollentyp", "Thema", "Verschluss", "Zwischensohlen-Typ"]

nlp = spacy.load("de_core_news_sm")

ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents":True})

ruler.add_patterns([{"label": "EU-Schuhgröße", "pattern": [{"TEXT": {"REGEX":"^EU$"}}, {"TEXT": {"REGEX":"^[3-5][0-9](,5)?$"}}]},
                    {"label": "EU-Schuhgröße", "pattern": [{"TEXT": {"REGEX":"^EU$"}}, {"TEXT": {"REGEX":"^[3-5][0-9](,5)?-[3-5][1-9](,5)?$"}}]},
                    {"label": "EU-Schuhgröße", "pattern": [{"TEXT": {"REGEX":"^EU[3-5][0-9](,5)?$"}}]},
                    {"label": "EU-Schuhgröße", "pattern": [{"TEXT": {"REGEX":"^(EU)?[3-5][0-9](,5)?-[3-5][0-9](,5)?$"}}]},
                    {"label": "EU-Schuhgröße", "pattern": [{"TEXT": {"REGEX":"^Gr\.[3-5][0-9](,5)?-[3-5][0-9](,5)?$"}}]},
                    {"label": "EU-Schuhgröße", "pattern": [{"TEXT": {"REGEX":"^Gr\.[3-5][0-9](,5)?$"}}]}])
ruler.add_patterns([{"label": "US-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^US$"}}, {"TEXT": {"REGEX":"^1?[0-9](,5)?$"}}]},
                   {"label": "US-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^US$"}}, {"TEXT": {"REGEX":"^1?[0-9](,5)?-1?[0-9](,5)?$"}}]},
                   {"label": "US-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^US1?[0-9](,5)?$"}}]},
                   {"label": "US-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^US1?[0-9](,5)?-1?[0-9](,5)?$"}}]}])
ruler.add_patterns([{"label": "UK-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^UK$"}}, {"TEXT": {"REGEX":"^1?[0-9](,5)?$"}}]},
                   {"label": "UK-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^UK$"}}, {"TEXT": {"REGEX":"^1?[0-9](,5)?-1?[0-9](,5)?$"}}]},
                   {"label": "UK-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^UK1?[0-9](,5)?$"}}]},
                   {"label": "UK-Schuhgröße", "pattern":[{"TEXT": {"REGEX":"^UK1?[0-9](,5)?-1?[0-9](,5)?$"}}]}])
ruler.add_patterns([{"label":"Erscheinungsjahr", "pattern":[{"TEXT": {"REGEX":"^20[0-2][0-9]$"}}]}])
ruler.add_patterns([{"label":"Herstellernummer", "pattern":[{"TEXT": {"REGEX":"^\d{5,}$"}}]},
                    {"label":"Herstellernummer", "pattern":[{"TEXT": {"REGEX":"^[A-Za-z1-9]{3,}-\d{2,}$"}}]},
                    {"label":"Herstellernummer", "pattern":[{"TEXT": {"REGEX":"^[A-Za-z]+\d+[A-Za-z1-9]{2,}$"}}]},])
ruler.add_patterns([{"label":"Maßeinheit", "pattern":[{"TEXT": {"REGEX":"^\d+(,\d+)?$"}}, {"LOWER": {"REGEX":"^cm|inches|pounds|grams|kg$"}}]}])

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="2023 eBay ML Challenge Data/Train_Tagged_Titles.tsv")
args = parser.parse_args()
os.makedirs("save", exist_ok=True)
data = pd.read_csv(args.data_path,sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
pre_record = 0
pre_tag = None
pattern = ""
ents = []
stack = []
for i, word in data.iterrows():
    
    if word["Record Number"]!=pre_record:
        pre_record = word["Record Number"]
        doc = nlp(word["Title"])
        
    if type(word["Tag"]) == float:
        pattern += " "
        pattern += word["Token"]
    else:
        if pre_tag in fixed_ent:
            if pattern in stack:
                ruler.add_patterns([{"label":pre_tag, "pattern":[{"LOWER":p.lower()} for p in pattern.split(" ")]}])
            else:
                stack.append(pattern)
        elif pre_tag == "Marke":
            if pattern in stack:
                ruler.add_patterns([{"label":"ORG", "pattern":[{"LOWER":p.lower()} for p in pattern.split(" ")]}])
            else:
                stack.append(pattern)
        pre_tag = word["Tag"]
        pattern = word["Token"]

ruler.to_disk("save/spacy_model")



