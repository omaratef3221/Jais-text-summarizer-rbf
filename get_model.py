import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from get_data import get_data_final
from datasets import Dataset
import time
import torch
import gc
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:47000"
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device, " is available")


def get_model(model_id):
    model_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                device_map="auto", 
                                                trust_remote_code=True, 
                                                offload_folder="offload")
    return tokenizer, model