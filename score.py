import os
import time
import yaml
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
df_test = pd.read_csv(cfg["paths"]["test_file"])

q_col = "question"
a_col = "expected_answer"

def load_model_and_tokenizer(model_path):
    tok = GPT2Tokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    mod.eval()
    return mod, tok

def generate_answer(model, tokenizer, question, use_chat_template=False):
    if use_chat_template:
        u_tok = cfg["special_tokens"]["user"]
        a_tok = cfg["special_tokens"]["assistant"]
        e_tok = cfg["special_tokens"]["end"]
        prompt = f"{u_tok} {question} {a_tok} "
        end_id = tokenizer.convert_tokens_to_ids(e_tok)
    else:
        prompt = f"Question: {question}\nAnswer:"
        end_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_t = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg["eval"]["max_new_tokens"],
            do_sample=False,
            eos_token_id=end_id,
            pad_token_id=tokenizer.pad_token_id
        )
    latency = time.time() - start_t

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return answer, latency

models_to_eval = [
    ("base", cfg["base_model"], False),
    ("cpt", cfg["paths"]["cpt_output"], False),
    ("sft", cfg["paths"]["sft_output"], True)
]

for tag, path, use_template in models_to_eval:
    print(f"Evaluating {tag} model...")
    model, tokenizer = load_model_and_tokenizer(path)
    records = []
    total_latency = 0.0

    for _, row in df_test.iterrows():
        q = str(row[q_col])
        expected = str(row[a_col])
        pred, lat = generate_answer(model, tokenizer, q, use_chat_template=use_template)
        total_latency += lat
        records.append({
            "question": q,
            "expected_answer": expected,
            "model_answer": pred,
            "score": "",
            "note": ""
        })

    out_csv = os.path.join(cfg["paths"]["results_dir"], f"results_{tag}.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    avg_lat = total_latency / len(df_test)
    print(f"Saved {out_csv} | Avg Latency: {avg_lat:.4f}s")
