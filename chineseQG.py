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

# Load fine tuned model and Tokenizer
model_save_path = "/data/local/cat_data/qgmodel2"
tokenizer_save_path = "/data/local/cat_data/qgmodel2"

model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
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

url = "https://github.com/Reagan1947/DRCD-simplified-Chinese/raw/master/"
data_files = {
    "train": url + "DRCD_traning_simplified_chinese.json",
    "test": url + "DRCD_dev_simplified_chinese.json",
}
drcd_raw_dataset = load_dataset("json", data_files=data_files, field = 'data',download_mode="force_redownload")

qac_pairs_train = []
for article in drcd_raw_dataset['train']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answer = qa['answers']
            qac_pairs_train.append({'question': question, 'answer': answer, 'context': context})

# Convert the qac_pairs list to a pandas DataFrame
df_train = pd.DataFrame(qac_pairs_train)

# Print the first few rows of the DataFrame
df_train.head(2)

qac_pairs_test = []
for article in drcd_raw_dataset['train']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answer = qa['answers']
            qac_pairs_test.append({'question': question, 'answer': answer, 'context': context})

# Convert the qac_pairs list to a pandas DataFrame
df_test = pd.DataFrame(qac_pairs_test)

# Print the first few rows of the DataFrame
df_test.head(2)

from datasets import Dataset

drcd_test = Dataset.from_pandas(df_test)
drcd_train = Dataset.from_pandas(df_train)

def add_eos_to_examples_zh(example):
    example['input_text'] = 'answer: %s  context: %s </s>' % (example['answer'][0]['text'], example['context'])
    example['target_text'] = '%s </s>' % example['question']
    return example

# tokenize the examples
def convert_to_features_zh(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, truncation=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, truncation=True, max_length=64)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

tokenized_dataset_zh  = drcd_train.map(add_eos_to_examples_zh)
tokenized_dataset_zh  = tokenized_dataset_zh.map(convert_to_features_zh, batched=True)

columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']

tokenized_dataset_zh = tokenized_dataset_zh.remove_columns(['question','answer','context','input_text','target_text'])

train_dataset_zh = tokenized_dataset_zh
train_dataset_zh.set_format(type='torch', columns=columns)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset_zh, collate_fn = T2TDataCollator(),batch_size = 5)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)


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

ft_model_save_path = "/data/local/cat_data/qgmodel_zh2"
ft_tokenizer_save_path = "/data/local/cat_data/qgmodel_zh2"

model.save_pretrained(ft_model_save_path)
tokenizer.save_pretrained(ft_tokenizer_save_path)