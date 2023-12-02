# Named Entity Recognition for German Commdity Titles
This is my implement for [eBay 2023 University Machine Learning Competition](https://eval.ai/web/challenges/challenge-page/2014/overview). I got an F1 Score of 0.930349 in quiz dataset, ranking 15 among all 65 submissions. 
This is a Named Entity Recognition (NER) task on Commdity Titles written in German, where we need to extract key information like Marke (Brand), Farbe (Color), Gewebeart (Fabric Type), etc. 
## Generate Entities
We will not share the dataset because of the Data Use Agreement. However, you can also try our model on your own data.  
### Data Format
Your data should be a csv or tsv file with a header of two columns: "Record Number" and "Title", where "Record Number" is an integer to identify the commdity, and "Title" is the commdity title from online shopping sites (should be in German, no further process needed).  
### Pretrained Models
You can download our pretrained model from [here](https://drive.google.com/file/d/1sd1wXN8nV1ScDpIXabthmuntvsbIv_PE/view?usp=sharing), unziping the archive file and move to the project folder.
### Generate
Run the following command:  
```
python result.py --data_path PATH_TO_YOUR_FILE
```
The generated file is "save/result.cvs", each name entity corresponding to a line, which contains 3 columns: first is the record number, second is entity label, third is entity content.  
Here is an example.   
Input:  
| Record Number | Title |
| ------ | ------ |
| 5001 | NIKE FREE RUN 3 SHIELD 5.0 SNEAKERS LAUFSCHUHE GR . EU 46 US 12 UK 11 30 CM |  
   
     
Output:  
| Record Number | Aspect Name | Aspect Value |
| ------ | ------ | ------ |
| 5001 | Marke | NIKE |
| 5001 | Produktlinie | FREE |
| 5001 | Modell | RUN 3 SHIELD 5.0 |
| 5001 | Stil | SNEAKERS |
| 5001 | Produktart | LAUFSCHUHE |
| 5001 | EU-Schuhgröße | EU 46 |
| 5001 | US-Schuhgröße | US 12 |
| 5001 | UK-Schuhgröße | UK 11 |
| 5001 | Maßeinheit | 30 CM |

## Training
In order to train the model, you need to get the dataset from eBay, and put it in the project folder. Or else, you may modify the code to fit in your own dataset.  

### Entity Ruler
To generate entity ruler, you need to run  
```
python spacy_ent.py
```
The entity ruler will be saved to "save/spacy_model".  

### Pretraining
Our dataset contains a huge amount of unlabeled data, which are in the "Listing_Titles.tsv" file, but only a small portion is labeled. Therefore, we pretrain a BERT model on the unlabeled data, then finetune the model on our task.
To remove all the special characters, run 
```
python pretrain_preprocess.py
```
The result file will be saved to "save/Listing_Titles.txt".
Then, to pretrain the BERT model, run  
```
python pretrain.py
```
The pretrained model will be saved to "save/mlm_model/pytorch_model.bin".

### Finetuning
To train the model on the full labeled dataset, run  
```
python train.py
```
If you want to valid the model to find the best hyperparameters, run
```
python train.py --valid True
```
