import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

print("GPU Ready:", torch.cuda.is_available())

model_path = "./checkpoints/cpt_final"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
special_tokens_dict = {'pad_token': '<|pad|>', 'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end|>']}
tokenizer.add_special_tokens(special_tokens_dict)

model = GPT2LMHeadModel.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

df = pd.read_csv("seed_test_set_30_rows.csv")
cols = df.columns.tolist()
q_col = [c for c in cols if 'question' in c.lower()][0]
a_col = [c for c in cols if 'answer' in c.lower()][0]

def process_data(example):
    question = str(example[q_col])
    answer = str(example[a_col])
    
    prompt_text = f"<|user|> {question} <|assistant|>"
    full_text = prompt_text + f" {answer} <|end|>"
    
    tokenized_full = tokenizer(full_text, truncation=True, max_length=256)
    tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=256)
    
    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    labels = input_ids.copy()
    
    prompt_len = len(tokenized_prompt["input_ids"])
    for i in range(prompt_len):
        labels[i] = -100
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

raw_dataset = Dataset.from_pandas(df)
tokenized_dataset = raw_dataset.map(process_data, remove_columns=raw_dataset.column_names)

training_args = TrainingArguments(
    output_dir="./checkpoints/sft_final_real",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    save_steps=15,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model("./checkpoints/sft_final_real")
tokenizer.save_pretrained("./checkpoints/sft_final_real")
print("SFT Training with -100 Loss Masking Completed!")
