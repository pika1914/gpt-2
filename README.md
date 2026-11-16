# PC Hardware Bottleneck LLM — Phase 2

**Team Members:**
* Elyas Hindi (Student ID: 182439)
* Omar Abu Khadair (Student ID: 162858)

## Model Checkpoints
Model weights (`gpt2-cpt` and `gpt2-sft`) are hosted externally on Google Drive:
* **Checkpoints Link:** [ضع رابط Google Drive هنا]

## Repository Structure
* `config.yaml`: All model names, paths, random seed (`42`), and training hyperparameters.
* `prepare_data.py`: Cleans raw CSV/JSON hardware data into CPT documents (90/10 split) and 380 SFT pairs.
* `train_cpt.py`: Stage 1 continued pretraining on GPT-2 small (`124M`) and perplexity evaluation.
* `train_sft.py`: Stage 2 supervised fine-tuning with custom chat tokens (`<|user|>`, `<|assistant|>`, `<|end|>`) and answer-only loss masking.
* `score.py`: Unified greedy-decoding evaluation script across `base`, `cpt`, and `sft` models.
* `SCOREBOARD.md`: Evaluation metrics across model stages.
* `results/`: Saved predictions (`results_*.csv`), CPT metrics, and training loss curves.

## How to Run
1. Install dependencies:
   ```bash
   pip install transformers datasets accelerate pyyaml pandas scikit-learn matplotlib
