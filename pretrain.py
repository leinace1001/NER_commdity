import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Step 1: Load a BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
model = BertForMaskedLM.from_pretrained("bert-base-german-cased")


model.bert.embeddings.requires_grad_ = False
for i in range(5):
  model.bert.encoder.layer[i].requires_grad_ = False
# Step 2: Prepare your text corpus
# Create a text file with your corpus and tokenize it
train_data_file = "save/Listing_Titles.txt"
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_data_file,
    block_size=128,  # You can adjust the block size based on your dataset
)

#torch.save(dataset, "save/pretrained_dataset")
#dataset = torch.load("save/pretrained_dataset")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,  # Adjust the probability of masking tokens
)

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir="./save/mlm_model",  # Output directory for the model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=32,  # Adjust based on your hardware
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./save/logs"
)

# Step 4: Create a Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start the training
trainer.train(resume_from_checkpoint=True)

# Save the pretrained model
#trainer.save_model()
torch.save(model.state_dict(),"save/mlm_model/pytorch_model.bin")