import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import TrainingArguments
from get_model import *
from get_data import *
import time
import requests
from add_rbf_to_model import replace_ffn_with_rbf_jais
from huggingface_hub.hf_api import HfFolder
from inference import *
from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM
import torch


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def main(args):
    tokenizer, model  = get_model(args.model_id)
    
    train_data = get_df(args.df_file_path, sample_size=10000)
    data = get_data_final(train_data)
    
    data = Dataset.from_pandas(data)
    
    training_output_dir = f'./JAIS_original_training-{str(int(time.time()))}'

    print("Number of Original Model parameters: ", print_number_of_trainable_model_parameters(model), flush=True)
    if args.EnableRBF == "rbf":
        replace_ffn_with_rbf_jais(model, 2)
        print("Number of RBF Model parameters: ", print_number_of_trainable_model_parameters(model), flush=True)
        device_map = infer_auto_device_map(model)  # max_memory can be specified if needed
        # model.parallelize(device_map)

        # model = AutoModelForCausalLM.from_config(
        #     model.config,  
        #     device_map=device_map 
        #     )
        print(model)
    
    training_params = TrainingArguments(
    output_dir=training_output_dir,
    save_strategy="steps",
    auto_find_batch_size=True,
    max_steps = -1,
    num_train_epochs=args.epochs,
    save_steps=1,
    logging_steps=500,
    learning_rate=1e-4,
    push_to_hub = True,
    # bf16 = True,
    hub_model_id = f"basic-jais-13b-arabic-text-summarizer",
    push_to_hub_model_id = f"basic-jais-13b-arabic-text-summarizer",
    # hub_token = args.token,
    )
    
    response_template = """
    ###
        التلخيص:   
        """
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    
    trainer = SFTTrainer(
    model,
    train_dataset=data,
    formatting_func=preprocess_dataset,
    data_collator=collator,
    )
    
    t1 = time.time()
    trainer.train()
    print("time taken: ", round((time.time() - t1), 5), flush=True)
    # trainer.save_model(f"./{args.model_id.split('/')[1]}")
    # trainer.save_tokenzer(f"./{args.model_id.split('/')[1]}")
    
    requests.post("https://ntfy.sh/master_experiment1", data="Experiment 1 Training Done".encode(encoding='utf-8'))
    
    trainer.push_to_hub(f"basic-jais-13b-arabic-text-summarizer",)
    tokenizer.push_to_hub(f"basic-jais-13b-arabic-text-summarizer",)
    
    requests.post("https://ntfy.sh/master_experiment1", data="Experiment 1 Model Uploaded to HuggingFace ".encode(encoding='utf-8'))
    
    requests.post("https://ntfy.sh/master_experiment1", data="Experiment 1 Done ".encode(encoding='utf-8'))
    
    print("Testing the model on test dataset starting....", flush=True)
    test_model_on_dataset(tokenizer, model)
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--df_file_path", type = str, default = "train.csv")
    parser.add_argument("--EnableRBF", type=str, default="rbf")
    # parser.add_argument("--token", type=str)
    args = parser.parse_args()
    # HfFolder.save_token(args.token)
    main(args)
    