import re
import spacy
from transformers import AutoTokenizer
from collections import defaultdict

Tags = ["No Tag", "Abteilung", "Aktivität", "Akzente", "Anlass", "Besonderheiten", "Charakter",
        "Charakter Familie", "Dämpfungsgrad", "Erscheinungsjahr", "EU-Schuhgröße", "Farbe", "Futtermaterial",
        "Gewebeart", "Herstellernummer", "Herstellungsland und-region", "Innensohlenmaterial", "Jahreszeit",
        "Laufsohlenmaterial", "Marke", "Maßeinheit", "Modell", "Muster", "Obermaterial", "Produktart",
        "Produktlinie", "Schuhschaft-Typ", "Schuhweite", "Stil", "Stollentyp", "Thema", "UK-Schuhgröße",
        "US-Schuhgröße", "Verschluss", "Zwischensohlen-Typ"]

spacy_ent = ["Abteilung", "Aktivität", "Akzente", "Anlass", "Besonderheiten", 
             "Dämpfungsgrad","Farbe", "Futtermaterial", "Gewebeart", "Innensohlenmaterial",
             "Jahreszeit", "Laufsohlenmaterial", "Muster", "Obermaterial", "Produktart", 
             "Schuhschaft-Typ", "Schuhweite", "Stil", "Stollentyp", "Thema", "Verschluss", "Zwischensohlen-Typ", 
             "LOC", "PER", "MISC", "ORG"]

Tags_map = {}
for i,t in enumerate(Tags):
    Tags_map[t] = i

Tags_map["Obscure"] = -100

spacy_ent_label = defaultdict(int)
for i,e in enumerate(spacy_ent):
    spacy_ent_label[e] = i+1

def my_split(s, sp): 
    """
    Split a string; return with position and a place for Tag
    """
    out = []
    last_pos = 0
    for i in range(len(s)):
        if s[i] == sp:
            out.append([s[last_pos:i], last_pos, None])
            last_pos = i+1
    out.append([s[last_pos:], last_pos, None])
    return out

def process_title(title, nlp):
    """
    To split the title into tokens of word; 
    Tag a token which consists of special characters and does not belong to any named entity (No Tag); 
    Tag the named entity recognized by pretrained spacy model with rule-based ruler
    """
    entity = my_split(title, " ")
    pe = 0 # pointer to the entity index
    ind = 0 # pointer to the string position
    
    doc = nlp(title)
    for ent in doc.ents:
        ind = title.find(ent.text, ind)
        while pe<len(entity) and entity[pe][1]<ind:
            pe += 1
        while pe<len(entity) and entity[pe][0] in ent.text:
            entity[pe][2] = ent.label_
            pe += 1
        ind += len(ent.text)
    for e in entity:
        if e[2] == None and re.match("^[^a-zA-Z0-9]+$", e[0]):
            e[2] = "No Tag"

    return entity
        
def process_cvs(data_frame): 
    """
    Fill all word without tag;
    Float nan if read by pandas
    """
    pre_tag = None
    for _, word in data_frame.iterrows():
        if type(word["Tag"]) == float:
            word["Tag"] = pre_tag
        else:
            pre_tag = word["Tag"]

def process_data(data_frame, nlp):
    pre_record = 0
    index = 0 #index of a token in a record
    data = []
    record = None
    entity = None
    tag = None
    for _, word in data_frame.iterrows():
        if word["Record Number"] != pre_record:
            if int(pre_record)>0:
                data.append(record)
            record = [[],[],[]]
            index = 0
            pre_record = word["Record Number"]
            
            entity = process_title(word["Title"], nlp)
        else:
            index += 1
        if entity[index][2]!="No Tag":
            record[0].append(word["Token"])
            if type(word["Tag"])!= float:
              tag = word["Tag"]
            if tag in Tags:
              record[2].append(Tags_map[tag])
            else:
              record[2].append(-100)
            if entity[index][2] == None:
                record[1].append(0)
            else:
                record[1].append(spacy_ent_label[entity[index][2]])
    data.append(record)
    return data
    
def tonkenize_aline(data, max_length = 80):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    inputs = []
    spacy_labels = []
    gt_labels = []
    word_ids = []
    for item in data:
        spacy_label = []
        gt_label = []
        word_id = []
        input = tokenizer(item[0],is_split_into_words=True, padding="max_length", max_length=max_length, return_tensors="pt")
        input['input_ids'] = input['input_ids'].view(-1)
        input['token_type_ids'] = input['token_type_ids'].view(-1)
        input['attention_mask'] = input['attention_mask'].view(-1)
        inputs.append(input)
        for i in input.word_ids():
            if i == None:
                spacy_label.append(0)
                gt_label.append(-100)
                word_id.append(-1)
            else:
                spacy_label.append(item[1][i])
                gt_label.append(item[2][i])
                word_id.append(i)
        spacy_labels.append(spacy_label)
        gt_labels.append(gt_label)
        word_ids.append(word_id)

    return inputs, spacy_labels, word_ids, gt_labels

if __name__ == "__main__":
    entity = process_title("adidas Originals Sneakers Damen Freizeitschuhe Turnschuhe Gr . UK 4 ( ... #e 32e049")
    print(entity)