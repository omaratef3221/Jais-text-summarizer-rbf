import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from Data_processing import get_data_final
from datasets import Dataset
import time
import torch
import gc
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:47000"
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "inception-mbzuai/jais-13b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map="auto", 
                                             trust_remote_code=True, 
                                             offload_folder="offload").to(device)

for param in model.parameters():
    param.requires_grad = True


model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3])

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model), flush=True)



def tokenize_function(example):
    start_prompt = 'قم بتلخيص هذه الفقرة: \n\n'
    end_prompt = '\n\nالتلخيص: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["text"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

data_df = get_data_final()
data = Dataset.from_pandas(data_df)

tokenized_datasets = data.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['summary', 'text',])
###### Training
gc.collect()
torch.cuda.empty_cache()
EPOCHS = 1
PRINT_INFO = True
LR = 1e-3
BATCH_SIZE = 2

training_output_dir = f'./JAIS_original_training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=training_output_dir,
    auto_find_batch_size=True,
    learning_rate=LR, 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=1,
    logging_strategy = 'epoch',
    max_steps=-1, 
)
    
trainer = Trainer(
    model=model,
    args=peft_training_args,
    train_dataset=tokenized_datasets,
)
start = time.time()
trainer.train()
print("Training time for Original model fine tuning: ", round(time.time() - start), " Seconds. ", flush = True)
new_model_path="./Jais_original_model-checkpoint-local"

trainer.model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)

trainer.push_to_hub("Jais_Summarizer_basic")
