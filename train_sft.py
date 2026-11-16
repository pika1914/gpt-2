import os
import json
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed
)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["seed"])

tokenizer = GPT2Tokenizer.from_pretrained(cfg["paths"]["cpt_output"])
special_tokens = {
    "additional_special_tokens": [
        cfg["special_tokens"]["user"],
        cfg["special_tokens"]["assistant"],
        cfg["special_tokens"]["end"]
    ]
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(cfg["paths"]["cpt_output"])
model.resize_token_embeddings(len(tokenizer))

with open(cfg["paths"]["sft_data"], "r", encoding="utf-8") as f:
    all_pairs = [json.loads(line) for line in f]

train_pairs, val_pairs = train_test_split(
    all_pairs, test_size=cfg["sft"]["val_split"], random_state=cfg["seed"]
)

class SFTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.samples = []
        u_tok = cfg["special_tokens"]["user"]
        a_tok = cfg["special_tokens"]["assistant"]
        e_tok = cfg["special_tokens"]["end"]

        for p in pairs:
            prompt_text = f"{u_tok} {p['question']} {a_tok} "
            answer_text = f"{p['answer']} {e_tok}"

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

            input_ids = (prompt_ids + answer_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]

            pad_len = max_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = SFTDataset(train_pairs, tokenizer, cfg["sft"]["max_length"])
val_dataset = SFTDataset(val_pairs, tokenizer, cfg["sft"]["max_length"])

training_args = TrainingArguments(
    output_dir=cfg["paths"]["sft_output"],
   
    num_train_epochs=cfg["sft"]["epochs"],
    per_device_train_batch_size=cfg["sft"]["batch_size"],
    per_device_eval_batch_size=cfg["sft"]["batch_size"],
    learning_rate=cfg["sft"]["learning_rate"],
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    seed=cfg["seed"],
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

trainer.save_model(cfg["paths"]["sft_output"])
tokenizer.save_pretrained(cfg["paths"]["sft_output"])

train_loss = [x["loss"] for x in trainer.state.log_history if "loss" in x]
plt.figure(figsize=(8, 4))
plt.plot(train_loss, label="SFT Train Loss (Answer Only)")
plt.title("SFT Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(cfg["paths"]["results_dir"], "sft_loss_curve.png"))
plt.close()

print("SFT training complete and saved.")
