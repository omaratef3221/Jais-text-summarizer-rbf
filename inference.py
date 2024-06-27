import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

def summarize_text(model, tokenizer, text, max_length=150):
    inputs = tokenizer.encode("قم بتلخيص النص التالي: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main():
    model_id = "omaratef3221/Qwen2-0.5B-Instruct-arabic-text-summarizer"
    tokenizer, model = load_model_and_tokenizer(model_id)
    
    text = """
    أدت الحرب العالمية الثانية إلى تغييرات عميقة في الحياة الاجتماعية والسياسية والاقتصادية في جميع أنحاء العالم.
    """
    
    summary = summarize_text(model, tokenizer, text)
    print("Original Text:\n", text)
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()
