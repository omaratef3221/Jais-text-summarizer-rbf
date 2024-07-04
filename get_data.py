import pandas as pd
import random
import re 
import nltk
nltk.download('punkt')

data_df = pd.read_csv("train.csv").head(200)

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


def get_data_final():
    data_df['text'] = data_df['text'].apply(text_prepare, args=(True,))
    data_df['summary'] = data_df['summary'].apply(text_prepare, args=(True,))
    return data_df

def preprocess_dataset(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f""" 
        قم بتلخيص النص التالي: {example["text"][i]} \n\n
        ###
        التلخيص: {example["summary"][i]}
        """
        output_texts.append(text)
    return output_texts