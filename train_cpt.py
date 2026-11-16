import os
import math
import yaml
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(cfg["base_model"])
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(cfg["base_model"]).to(device)

dataset = load_dataset("text", data_files={
    "train": cfg["paths"]["cpt_train"],
    "validation": cfg["paths"]["cpt_val"]
})

def tokenize_fn(examples):
    return tokenizer(examples["text"])

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
block_size = cfg["cpt"]["block_size"]

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=cfg["paths"]["cpt_output"],
    
    num_train_epochs=cfg["cpt"]["epochs"],
    per_device_train_batch_size=cfg["cpt"]["batch_size"],
    per_device_eval_batch_size=cfg["cpt"]["batch_size"],
    learning_rate=cfg["cpt"]["learning_rate"],
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    seed=cfg["seed"],
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

eval_before = trainer.evaluate()
ppl_before = math.exp(eval_before["eval_loss"])

trainer.train()

eval_after = trainer.evaluate()
ppl_after = math.exp(eval_after["eval_loss"])

trainer.save_model(cfg["paths"]["cpt_output"])
tokenizer.save_pretrained(cfg["paths"]["cpt_output"])

train_loss = [x["loss"] for x in trainer.state.log_history if "loss" in x]
eval_loss = [x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x]

plt.figure(figsize=(8, 4))
plt.plot(train_loss, label="Train Loss")
plt.title("CPT Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(cfg["paths"]["results_dir"], "cpt_loss_curve.png"))
plt.close()

with open(os.path.join(cfg["paths"]["results_dir"], "cpt_metrics.txt"), "w") as f:
    f.write(f"Perplexity Before CPT: {ppl_before:.2f}\n")
    f.write(f"Perplexity After CPT: {ppl_after:.2f}\n")
    f.write(f"Total Train Tokens: {len(lm_datasets['train']) * block_size}\n")

print(f"Perplexity Before: {ppl_before:.2f} -> After: {ppl_after:.2f}")
