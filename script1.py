import numpy as np
import opendatasets as od
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from Data_processing import get_data_final
from datasets import Dataset


import time

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

train=data_df.sample(frac=0.7,random_state=7) # Create training of 70% of the data
test=data_df.drop(train.index) # Create testing by removing the 70% of the train data which will result in 30%

val=test.sample(frac=0.5,random_state=7) # Create validation of 50% of the testing data
test=test.drop(val.index) # Create testing by removing the 50% of the validation data which will result in 50%

test.to_csv("testing_data.csv", index = None)

print("Training Shape: ", train.shape) # Print the Trainnig shape (rows, columns)
print("Validation Shape: ", val.shape) # Print the Validation shape (rows, columns)
print("Testing Shape: ", test.shape) # Print the Testing shape (rows, columns)

def tokenize_function(example):
    start_prompt = 'قم بتلخيص هذه الفقرة: \n\n'
    end_prompt = '\n\nالتلخيص: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["text"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example


# Train Data
train_data = Dataset.from_pandas(train)
train_tokenized_datasets = train_data.map(tokenize_function, batched=True)
train_tokenized_datasets = train_tokenized_datasets.remove_columns(['summary', 'text',])


# Val Data
val_data = Dataset.from_pandas(val)
val_tokenized_datasets = val_data.map(tokenize_function, batched=True)
val_tokenized_datasets = val_tokenized_datasets.remove_columns(['summary', 'text',])


# Test Data
test_data = Dataset.from_pandas(test)
test_tokenized_datasets = test_data.map(tokenize_function, batched=True)
test_tokenized_datasets = test_tokenized_datasets.remove_columns(['summary', 'text',])

EPOCHS = 25
LR = 1e-4
BATCH_SIZE = 1

training_output_dir = f'./JAIS_original_training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=training_output_dir,
    learning_rate=LR,
    num_train_epochs=25,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    logging_steps=1,
    logging_strategy = 'epoch',
    max_steps=-1,
    bf16 = True,
    push_to_hub = True,
    hub_model_id = 'JAIS_Text_Summarizer_Basic_13B',
    hub_token = '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',
    # deepspeed="deep_speed_config.json"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_tokenized_datasets,
    # eval_dataset = val_tokenized_datasets
)

start = time.time()
trainer.train()
print("Training time for Original model fine tuning: ", round(time.time() - start), " Seconds. ", flush = True)

trainer.push_to_hub()