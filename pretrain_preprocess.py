import pandas as pd
import re
import os
from data_tool import *
from csv import QUOTE_NONE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="2023 eBay ML Challenge Data/Listing_Titles.tsv")
args = parser.parse_args()
os.makedirs("save", exist_ok=True)
raw_data = pd.read_csv(args.data_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=QUOTE_NONE)
titles = []
for i, r in raw_data.iterrows():
    
    title = ""
    l = r["Title"].split(" ")
    for token in l:
        if re.match("^[^a-zA-Z0-9]+$", token):
            pass
        else:
            title+=token
            title+=" "
    print(r["Record Number"])
    titles.append(title[:-1]+"\n")

with open("save/Listing_Titles.txt", "w") as f:
    for title in titles:
        f.write(title)