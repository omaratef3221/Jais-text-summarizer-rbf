import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from get_data import *
import time
import requests

data_df = pd.read_csv('test.csv').dropna(inplace = False)
data_df = data_df.head(3) #600

print("Test Data Shape: ", data_df.shape)

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
                                                trust_remote_code=True, 
                                                offload_folder="offload", torch_dtype = torch.bfloat16)
    return tokenizer, model

def summarize_text(model, tokenizer, text):
    prompt = f""" 
        قم بتلخيص النص التالي: {text} \n\n
        ###
        التلخيص: 
        """
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
    outputs = model.generate(inputs, max_length=100)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main(args):
    model_id = args.model_id
    tokenizer, model = load_model_and_tokenizer(model_id)
    
    data_df['text'] = data_df['text'].apply(text_prepare, args=(True,))
    
    t = time.time()
    data_df["model_summary"] = data_df["text"].apply(lambda x: summarize_text(model, tokenizer, x))
    print("Inference time taken for 600 rows was: ", round((time.time() - t), 5), " seconds", flush = True)
    
    data_df.to_csv(args.output_file_name, index=False)    
    requests.post("https://ntfy.sh/master_experiment1", data="Inference of Experiment 1 is done ".encode(encoding='utf-8'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_file_name", type=str)
    args = parser.parse_args()
    main(args)
