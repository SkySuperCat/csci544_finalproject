from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, MT5ForConditionalGeneration, T5TokenizerFast

from tqdm import tqdm

from typing import Dict, List, Optional

import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch


from huggingface_hub import notebook_login

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    EvalPrediction,
    DataCollator,
    Trainer,
    TrainingArguments)

raw_dataset = load_dataset("squad")

raw_dataset["train"][0]

"""# Preprocessing"""

checkpoint = "google/mt5-base"
#model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)

def add_eos_to_examples(example):
    example['input_text'] = 'answer: %s  context: %s </s>' % (example['answers']['text'][0], example['context'])
    example['target_text'] = '%s </s>' % example['question']
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, truncation=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, truncation=True, max_length=64)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

tokenized_dataset  = raw_dataset.map(add_eos_to_examples)
tokenized_dataset  = tokenized_dataset.map(convert_to_features, batched=True)

columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']

tokenized_dataset = tokenized_dataset.remove_columns(['id','title','context','question','answers','input_text','target_text'])
#valid_dataset = valid_dataset.remove_columns(['id','title','context','question','answers','input_text','target_text'])

train_dataset = tokenized_dataset['train']
valid_dataset = tokenized_dataset['validation']

train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

train_dataset_small = train_dataset.select(range(100))
valid_dataset_small = valid_dataset.select(range(10))

@dataclass
class T2TDataCollator():
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }



model = MT5ForConditionalGeneration.from_pretrained(checkpoint)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, collate_fn = T2TDataCollator(),batch_size = 5)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

import torch

device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()        
        optimizer.zero_grad()
        lr_scheduler.step()
        progress_bar.update(1)


model_save_path = "/data/local/cat_data/qgmodel2"
tokenizer_save_path = "/data/local/cat_data/qgmodel2"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)