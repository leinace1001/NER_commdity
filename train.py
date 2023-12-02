import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os.path as osp
import os
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from csv import QUOTE_NONE

from dataset import *
from model import *

def calculate_valid_loss(model, dataloader, device):
  with torch.no_grad():
    precise_history = []
    for _, data in enumerate(dataloader):
        inputs, spacy_label, word_ids, gt_label, _ = data
        inputs = inputs.to(device)
        spacy_label = spacy_label.to(device)
        word_ids = word_ids.to(device)
        gt_label = gt_label.to(device)
        
        output = model(inputs, word_ids, spacy_label)
        pd_label = output.argmax(dim=2)
        
        num_correct = torch.sum((gt_label==pd_label).int(), dim=1)
        num_total = torch.sum((gt_label!=-100).int(), dim=1)
        precise = torch.mean(num_correct/num_total)
        precise_history.append(precise.item())

  return sum(precise_history)/len(precise_history)
    

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=50)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--outdir", default="save/train")
parser.add_argument("--save_per_iter", default=1000)
parser.add_argument("--log_per_iter", default=100)
parser.add_argument("--hidden_dim", default=256)
parser.add_argument("--input", default="2023 eBay ML Challenge Data/Train_Tagged_Titles.tsv")
parser.add_argument("--lr", default=1e-5)
parser.add_argument("--pretrained", default="save/mlm_model/pytorch_model.bin")
parser.add_argument("--pseudo_label", default=False)
parser.add_argument("--valid", default=False)
parser.add_argument("--resume")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs(args.outdir, exist_ok=True)
dataset = LabeledTitle(args.input)
print(len(dataset))
trainset = Subset(dataset, [i for i in range(4500)])
validset = Subset(dataset, [i for i in range(4500, 5000)])
if args.valid:
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
else:
    trainset = dataset
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(validset, batch_size=args.batch_size)

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
bert = BertForMaskedLM.from_pretrained("bert-base-german-cased")

bert.load_state_dict(torch.load(args.pretrained))
bert = bert.bert
model = NERModel(64, 35, bert).to(device)
model.train()
model.freeze()
titles = pd.read_csv("2023 eBay ML Challenge Data/Listing_Titles.tsv", sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=QUOTE_NONE).iloc[5000:]

optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)

best_precise = 0
it = 0
for e in range(args.epoch):
    if e>5 and e%2 == 0:
        sub_titles_ = titles.sample(2000)
        sub_titles = []
        for _, r in sub_titles_.iterrows():
            sub_titles.append(r["Title"])
        persudoset = PseudoLabelTitle(sub_titles, tokenizer, model, device)
        combinedset = ConcatDataset([persudoset, trainset])
        train_loader = DataLoader(combinedset, batch_size=args.batch_size, shuffle=True)
    for _, data in enumerate(train_loader):
        inputs, spacy_label, word_ids, gt_label, reg = data
        inputs = inputs.to(device)
        spacy_label = spacy_label.to(device)
        word_ids = word_ids.to(device)
        gt_label = gt_label.to(device)
        reg = reg.to(device)
        output = model(inputs, word_ids, spacy_label, True)
        prob = F.softmax(output, dim=2)
        entropy_reg = (prob*torch.log(prob)).sum(dim=2).mean(dim=1)
        
        loss = F.cross_entropy(output.view(-1, 35), gt_label.view(-1), label_smoothing=0.1) + 0.1 * (reg*entropy_reg).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it%args.log_per_iter == 0:
            valid_precise = calculate_valid_loss(model, valid_loader, device)
            print("iter {}   train loss {}   validation precise {}".format(it, loss.item(), valid_precise))
            if valid_precise > best_precise:
                best_precise = valid_precise
                path = osp.join(args.outdir, "best.pth".format(it))
                torch.save(model.state_dict(), path)
        
        if it%args.save_per_iter == 0:
            path = osp.join(args.outdir, "last.pth".format(it))
            torch.save(model.state_dict(), path)
        torch.cuda.empty_cache()
        it += 1

torch.save(model.state_dict(), osp.join(args.outdir, "final.pth"))