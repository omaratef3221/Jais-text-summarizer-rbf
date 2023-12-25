import numpy as np
import opendatasets as od
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from Data_processing import get_data_final
from datasets import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import time

# Initialize distributed training
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

torch.set_default_dtype(torch.bfloat16)

od.download('https://www.kaggle.com/datasets/fadyelkbeer/arabic-text-summarization-30-000')

data_df = get_data_final()
print(data_df.shape)

model_path = "inception-mbzuai/jais-13b"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             offload_folder="offload",
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)

# Prepare Data
train = data_df.sample(frac=0.7, random_state=7)
test = data_df.drop(train.index)
val = test.sample(frac=0.5, random_state=7)
test = test.drop(val.index)

# Tokenization
def tokenize_function(example):
    start_prompt = 'قم بتلخيص هذه الفقرة: \n\n'
    end_prompt = '\n\nالتلخيص: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["text"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

train_data = Dataset.from_pandas(train).map(tokenize_function, batched=True).remove_columns(['summary', 'text'])
val_data = Dataset.from_pandas(val).map(tokenize_function, batched=True).remove_columns(['summary', 'text'])

# Create distributed samplers
train_sampler = DistributedSampler(train_data)
val_sampler = DistributedSampler(val_data)

EPOCHS = 25
LR = 1e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2  # Adjust as needed

training_output_dir = f'./JAIS_original_training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=training_output_dir,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=1,
    logging_strategy='epoch',
    max_steps=-1,
    bf16=True,
    push_to_hub=True,
    hub_model_id='JAIS_Text_Summarizer_Basic_13B',
    hub_token='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',
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'labels': torch.stack([f['labels'] for f in data])},
    train_sampler=train_sampler,
    eval_sampler=val_sampler
)

start = time.time()
trainer.train()
print("Training time for Original model fine-tuning: ", round(time.time() - start), " Seconds.", flush=True)

if local_rank == 0:
    trainer.push_to_hub()
