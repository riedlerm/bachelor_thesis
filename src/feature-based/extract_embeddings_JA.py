#!/usr/bin/env python
# coding: utf-8


import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from transformers import BertTokenizer, BertForSequenceClassification, Adafactor
from transformers.optimization import AdafactorSchedule
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


language = 'ja'
num_labels = 5

raw_datasets = load_dataset('amazon_reviews_multi', language)
raw_datasets = raw_datasets.remove_columns(['review_id', 'product_id', 'reviewer_id', 'product_category'])

model_checkpoints = {'de' : 'bert-base-german-cased',
                     'en' : 'bert-base-uncased',
                     'ja' : 'cl-tohoku/bert-base-japanese',
                     'zh' : 'bert-base-chinese'}

# Hyperparameters

num_train_epochs = 2
learning_rate = 2e-5
seed = 2
gradient_accumulation_steps = 4


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


tokenizer = BertTokenizer.from_pretrained(model_checkpoints[language], use_fast=True)


max_len = 256
pad_to_max = False
def tokenize_function(example):
    text_ = example['review_body'] + " " + example['review_title'] 
    encodings = tokenizer.encode_plus(text_, pad_to_max_length=pad_to_max, max_length=max_len,
                                          add_special_tokens=True,
                                          return_attention_mask=True,
                                          )

    targets = example['stars']-1
    encodings.update({'labels': targets}) 
    return encodings


tokenized_datasets = raw_datasets.map(tokenize_function)
tokenized_datasets.column_names
tokenized_datasets = tokenized_datasets.remove_columns(["stars", "review_body", "review_title", "language"])
tokenized_datasets.column_names
tokenized_datasets.set_format("torch")



def pad_seq(seq, max_batch_len, pad_value):
    return seq.tolist() + (max_batch_len - len(seq)) * [pad_value]


@dataclass
class SmartCollator():
    pad_token_id: int

    def __call__(self, batch):
        batch_inputs = list()
        batch_attention_masks = list()
        labels = list()
        max_size = max([len(ex['input_ids']) for ex in batch])
        for item in batch:
            batch_inputs += [pad_seq(item['input_ids'], max_size, self.pad_token_id)]
            batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
            labels.append(item['labels'])

        return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
                }



data_collator = SmartCollator(pad_token_id=tokenizer.pad_token_id)
batch_size = 8



train_dataloader = DataLoader(
  tokenized_datasets["train"] , shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
  tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
)
test_dataloader = DataLoader(
  tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator
)



for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})


model = BertForSequenceClassification.from_pretrained(model_checkpoints[language], num_labels=5, output_hidden_states = True)


outputs = model(**batch, output_hidden_states=True)
print(outputs.loss, outputs.logits.shape, outputs.hidden_states[-2].shape)


optimizer = Adafactor(model.parameters(), 
                      scale_parameter=False, 
                      relative_step=False, 
                      warmup_init=False, 
                      lr=learning_rate)


num_epochs = num_train_epochs
num_training_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)



train_progress_bar = tqdm(range(num_training_steps))

all_train_embeds = []
train_labels = []

training_stats = []
global_step = 0

metric= load_metric("accuracy")
accuracy = {}

    
    
accelerator = Accelerator(fp16=True)
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

best_accuracy = float('-inf')


for epoch in range(num_epochs):
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')
    
    train_loss = 0
    
    model.train()
    for step, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_hidden_states=True)
        
        loss = outputs.loss
        loss = loss/gradient_accumulation_steps
        train_loss += loss.item()
        
        hidden_states = outputs.hidden_states

        accelerator.backward(loss)
        
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
            global_step += 1
            train_progress_bar.update(1)
            
            
            # evaluate every 500 steps and save stats

            if global_step % 500 == 0 and not global_step == 0:

                print(global_step)


                print("")
                print("Running Validation...")

                eval_progress_bar = tqdm(range(len(eval_dataloader)))

                eval_loss = 0

                model.eval()
                eval_loss = 0
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    loss = outputs.loss
                    eval_loss += loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
                    eval_progress_bar.update(1)


                avg_eval_loss = eval_loss/len(eval_dataloader)
                eval_metric = metric.compute()
                print(eval_metric)



                training_stats.append(
                    {
                        'Epoch': epoch + 1,
                        'Training Loss': avg_train_loss,
                        'Validation Loss': avg_eval_loss,
                        'Validation Accuracy': eval_metric["accuracy"],
                        'Step': global_step
                    }
                )   

            
        avg_train_loss = train_loss/len(train_dataloader)  
            
            
        i = 0
        # go through each sentence at the second from last layer:
        while i < len(hidden_states[-2]):
            # following code to get the sentence embedding from the CLS (first token of each sentence)
            sentence_embedding = hidden_states[-2][i][0]
            # add the embeding to the list of sentence embeddings 
            all_train_embeds.append(sentence_embedding.cpu())
            i += 1
            
        train_labels.extend(batch["labels"].tolist())
                    
         
        
print("")
print("Training complete!")  
model.save_pretrained(f'Models/Monolingual/Training/training_no_trainer_{language}')
print("Saving model...")



all_test_embeds = []
test_labels = []


print("")
print("Running Prediction...")

eval_progress_bar = tqdm(range(len(test_dataloader)))

eval_loss = 0


metric= load_metric("accuracy")
model.eval()
eval_loss = 0
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    loss = outputs.loss
    hidden_states = outputs.hidden_states
    eval_loss += loss.item()
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    eval_progress_bar.update(1)

    test_labels.extend(batch["labels"].tolist())

    i = 0
    while i < len(hidden_states[-2]):
        sentence_embedding = hidden_states[-2][i][0]
        all_test_embeds.append(sentence_embedding.cpu())
        i += 1


avg_eval_loss = eval_loss/len(eval_dataloader)
eval_metric = metric.compute()
print(eval_metric)




df_stats = pd.DataFrame(data=training_stats)
print(df_stats)
df_stats.to_csv(f"Models/Monolingual/Embeddings/stats_{language}.csv", index=False)




sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Validation Loss'], 'r--o', label="Validation")

plt.title("Training & Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'Models/Monolingual/Embeddings/stats_{language}.pdf')




df_stats = df_stats.set_index('Epoch')
df_stats.loc[df_stats['Validation Accuracy'].idxmax()]
best_accuracy_epoch = df_stats['Validation Accuracy'].idxmax()
print(best_accuracy_epoch)



size = (len(all_train_embeds)/num_epochs)
print(size)

all_train_embeds = all_train_embeds[int(size):]
train_labels = train_labels[int(size):]


df_train_embeds = pd.DataFrame()
for instance in all_train_embeds:
    numpy_instance = instance.detach().numpy()
    reshaped_numpy = np.reshape(numpy_instance, (1,768))
    numpy_df = pd.DataFrame(reshaped_numpy)
    df_train_embeds = pd.concat([df_train_embeds, numpy_df], ignore_index=True)


df_train_embeds.insert(0,"gold_label",train_labels)
df_train_embeds
df_train_embeds.to_csv(f"Models/Monolingual/Embeddings/train_embeddings_{language}.csv", index=False)



df_test_embeds = pd.DataFrame()
for instance in all_test_embeds:
    numpy_instance = instance.detach().numpy()
    reshaped_numpy = np.reshape(numpy_instance, (1,768))
    numpy_df = pd.DataFrame(reshaped_numpy)
    df_test_embeds = pd.concat([df_test_embeds, numpy_df], ignore_index=True)



df_test_embeds.insert(0,"gold_label",test_labels)
df_test_embeds
df_test_embeds.to_csv(f"Models/Monolingual/Embeddings/test_embeddings_{language}.csv", index=False)
