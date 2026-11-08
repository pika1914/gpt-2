import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback

print("GPU Ready:", torch.cuda.is_available())

os.makedirs("data/cpt", exist_ok=True)
train_file = "data/cpt/train.txt"
val_file = "data/cpt/val.txt"

if not os.path.exists(train_file):
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("JUST B.Sc. in AI Study Plan requirements and hardware guides.\n" * 100)
if not os.path.exists(val_file):
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("JUST Undergraduate Academic Regulations and rules.\n" * 20)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=512)
val_dataset = TextDataset(tokenizer=tokenizer, file_path=val_file, block_size=512)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./checkpoints/cpt_final",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()
trainer.save_model("./checkpoints/cpt_final")
tokenizer.save_pretrained("./checkpoints/cpt_final")
print("CPT Training Completed and Model Saved!")
