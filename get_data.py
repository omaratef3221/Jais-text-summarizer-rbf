import pandas as pd
import random
import re 
import torch
import nltk
nltk.download('punkt')


counter = 0
def delete_links(input_text):
    pettern  = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    out_text = re.sub(pettern, ' ', input_text)
    return out_text

def delete_repeated_characters(input_text):
    pattern  = r'(.)\1{2,}'
    out_text = re.sub(pattern, r"\1\1", input_text)
    return out_text

def remove_extra_spaces(input_text):
    replace = ' +'
    out_text = re.sub(replace, " ", input_text)
    words = nltk.word_tokenize(out_text)
    words = [word for word in words if word.isalpha()]
    out_text = ' '.join(words)
    return out_text

def replace_letters(input_text):
    replace = {"أ": "ا","ة": "ه","إ": "ا","آ": "ا","": ""}
    replace = dict((re.escape(k), v) for k, v in replace.items()) 
    pattern = re.compile("|".join(replace.keys()))
    out_text = pattern.sub(lambda m: replace[re.escape(m.group(0))], input_text)
    return out_text

def clean_text(input_text):
    replace = r'[^\u0621-\u064A\u0660-\u0669\u06F0-\u06F90-9]'
    out_text = re.sub(replace, " ", input_text)
    words = nltk.word_tokenize(out_text)
    #words = [word for word in words if word.isalpha()]
    out_text = ' '.join(words)
    return out_text

def remove_vowelization(input_text):
    vowelization = re.compile(""" ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ""", re.VERBOSE)
    out_text = re.sub(vowelization, '', input_text)
    return out_text

def delete_stopwords(input_text):
    stop_words = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    wnl = nltk.WordNetLemmatizer()
    lemmatizedTokens =[wnl.lemmatize(t) for t in tokens]
    out_text = [w for w in lemmatizedTokens if not w in stop_words]
    out_text = ' '.join(out_text)
    return out_text

def stem_text(input_text):
    st = ISRIStemmer()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    out_text = [st.stem(w) for w in tokens]
    out_text = ' '.join(out_text)
    return out_text

def text_prepare(input_text, ar_text):
    global counter
    counter +=1

    #out_text = delete_links(input_text)
    #out_text = delete_repeated_characters(out_text)
    #out_text = delete_stopwords(input_text)
    out_text = clean_text(input_text)
    #out_text = remove_extra_spaces(out_text)
    # if(counter%100==0):
    #     print(counter,'\n',out_text)
    return out_text

def get_df(df_path, sample_size = 10000):
    data = pd.read_csv(df_path).sample(sample_size)
    data.dropna()
    return data

def get_data_final(df):
    df['text'] = df['text'].apply(text_prepare, args=(True,))
    df['summary'] = df['summary'].apply(text_prepare, args=(True,))
    return df

def preprocess_dataset(examples, tokenizer):
    input_texts = []
    output_texts = []
    
    # Iterate over the input and summary (label) texts
    for i in range(len(examples['text'])):
        # Prepare the input text by adding the task-specific prompt
        input_text = f"قم بتلخيص النص التالي: {examples['text'][i]} \n\n"
        # The expected output text (i.e., the summary)
        output_text = f"التلخيص: {examples['summary'][i]}"
        
        input_texts.append(input_text)
        output_texts.append(output_text)

    # Tokenize the inputs and labels separately, applying truncation and max_length
    inputs = tokenizer(
        input_texts, 
        max_length=1024, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    labels = tokenizer(
        output_texts, 
        max_length=50,  # Adjust based on your task
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    # Replace padding token IDs in labels with -100 to ignore during loss computation
    labels["input_ids"] = labels["input_ids"].masked_fill(labels["input_ids"] == tokenizer.pad_token_id, -100)

    # Ensure inputs and labels are returned with correct dimensions
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }
