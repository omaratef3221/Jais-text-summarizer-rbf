import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from get_data import *
import time
import requests



def summarize_text(model, tokenizer, text):
    prompt = f""" 
        قم بتلخيص النص التالي: {text} \n\n
        ###
        التلخيص: 
        """
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=50)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def test_model_on_dataset(tokenizer, model):
    data_df = pd.read_csv('test.csv').dropna(inplace = False)
    data_df = data_df.head(500) #600

    print("Test Data Shape: ", data_df.shape)
    
    data_df['text'] = data_df['text'].apply(text_prepare, args=(True,))
    
    t = time.time()
    data_df["model_summary"] = data_df["text"].apply(lambda x: summarize_text(model, tokenizer, x))
    print("Inference time taken for 500 rows was: ", round((time.time() - t), 5), " seconds", flush = True)
    
    data_df.to_csv("Testing_Results.csv", index=False)    
    requests.post("https://ntfy.sh/master_experiment1", data="Inference of Experiment 1 is done ".encode(encoding='utf-8'))
    
