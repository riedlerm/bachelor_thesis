#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from datasets import load_dataset, load_metric
from dataclasses import dataclass
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from transformers.utils import logging



test_language = 'de'
num_labels = 5



"""## Loading the dataset and removing unused columns"""


test_dataset = load_dataset('amazon_reviews_multi', test_language, split='test')
test_dataset = test_dataset.remove_columns(['review_id', 'product_id', 'reviewer_id', 'product_category'])



model_checkpoint = "Models/MonoTrans/Training/finetune_model_tokenizer_DE"
print("Loading finetuned model from checkpoint:\n" + model_checkpoint)


tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

print(model.get_input_embeddings())


# Extract token embeddings of target language (test_langauge)

target_model_checkpoint = "Models/MonoTrans/Training/finetune_embeddings_tokenizer_DE"
target_model = BertForSequenceClassification.from_pretrained(target_model_checkpoint, num_labels=num_labels)


embeddings = target_model.get_input_embeddings()
print(embeddings)


# replace word embeddings of finetuned model with target embeddings

model.set_input_embeddings(embeddings)



max_len = 256
pad_to_max = False
def tokenize_data(example):
    text_ = example['review_body'] + " " + example['review_title'] 
    encodings = tokenizer.encode_plus(text_, pad_to_max_length=pad_to_max, max_length=max_len,
                                           add_special_tokens=True,
                                            return_attention_mask=True,
                                           )
    
    targets = example['stars']-1
    encodings.update({'labels': targets})
    return encodings


encoded_test_dataset = test_dataset.map(tokenize_data)



def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]


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



batch_size = 8

metric = load_metric('accuracy')


args = TrainingArguments(
    output_dir = "Models/MonoTrans/Training/monotrans_EN_DE_eval",
    seed = 42, 
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    eval_steps = 500,
    save_steps = 500,
    save_total_limit = 1,
    eval_accumulation_steps = 5,
    fp16 = True,
    optim="adafactor",
    gradient_accumulation_steps=2,
    gradient_checkpointing= True,

)



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
 
    predictions = np.argmax(predictions, axis=1)

    return metric.compute(predictions=predictions, references=labels)



trainer = Trainer(
    model,
    args=args,
    eval_dataset=encoded_test_dataset,
    data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



test_results = trainer.predict(encoded_test_dataset)
print(test_results.metrics)
print("Test accuracy:", test_results.metrics["test_accuracy"])

