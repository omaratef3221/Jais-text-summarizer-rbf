import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import argparse
from get_data import *

data = pd.read_csv('test.csv').dropna(inplace = False)
data = data.head(600)

print("Test Data Shape: ", data.shape)

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

def summarize_text(model, tokenizer, text):
    inputs = tokenizer.encode("قم بتلخيص النص التالي: " + text, return_tensors="pt")
    outputs = model.generate(inputs, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main(args):
    model_id = args.model_id
    tokenizer, model = load_model_and_tokenizer(model_id)
    
    data_df['text'] = data_df['text'].apply(text_prepare, args=(True,))
    
    
    data_df["model_summary"] = data_df["text"].apply(lambda x: summarize_text(model, tokenizer, x))
    
    data_df.to_csv(args.output_file_name, index=False)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_file_name", type=str)
    args = parser.parse_args()
    main(args)
