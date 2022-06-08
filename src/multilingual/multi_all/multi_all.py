# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from datasets import load_dataset, load_metric
from dataclasses import dataclass
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from transformers.utils import logging


logging.set_verbosity(40)

test_language = 'zh'

do_train = True


path = "Models/Multilingual/Training/"
directory = os.path.basename(__file__)
filename = os.path.splitext(directory)[0]


"""## Loading the dataset and removing unused columns and rows"""


dataset = load_dataset('amazon_reviews_multi')

if not do_train:
    dataset['test'] = load_dataset('amazon_reviews_multi', test_language, split='test')
    
dataset = dataset.remove_columns(['review_id', 'product_id', 'reviewer_id', 'product_category'])

languages = ["de", "en", "ja", "zh"]
dataset = dataset.filter(lambda example : example["language"] in languages)



"""## Preprocessing the data"""

do_shard = False
if do_shard:
    dataset = dataset.shuffle(seed=123)
    train_dataset = dataset["train"].shard(index=1, num_shards=100) 
    val_dataset = dataset['validation'].shard(index=1, num_shards=20) 
else:
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    
    
resume_training = False

if do_train:
    if resume_training:
        model_checkpoint = path + filename + '/checkpoint-3000'
    else:
        model_checkpoint = 'bert-base-multilingual-cased'
else:
    model_checkpoint = "Models/Multilingual/Training/multi_all"
    print("=====================")
    print("Loading finetuned model from checkpoint:\n" + model_checkpoint)


tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)


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


encoded_train_dataset = train_dataset.map(tokenize_data)
encoded_val_dataset = val_dataset.map(tokenize_data)
encoded_test_dataset = test_dataset.map(tokenize_data)



"""## Fine-tuning the model"""

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
num_labels = 5


metric = load_metric('accuracy')
metric_name = "accuracy"


args = TrainingArguments(
    output_dir = path + filename,
    seed = 127, 
    evaluation_strategy = "steps",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    eval_steps = 500,
    save_steps = 500,
    save_total_limit = 1,
    eval_accumulation_steps = 5,
    fp16 = True,
    optim="adafactor",
    gradient_accumulation_steps=4,
    gradient_checkpointing= True

)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
 
    predictions = np.argmax(predictions, axis=1)

    return metric.compute(predictions=predictions, references=labels)


model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)



trainer = Trainer(
    model,
    args=args,
    train_dataset= encoded_train_dataset, 
    eval_dataset=encoded_val_dataset if do_train else encoded_test_dataset,
    data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


if do_train:
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    
    
else:
    test_results = trainer.predict(encoded_test_dataset)
    print(test_results.metrics)
    print("Test accuracy:", test_results.metrics["test_accuracy"])

