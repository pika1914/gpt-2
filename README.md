# gpt-2
# BuildMate / JUST-AI Advisor - Phase 2
**Repository Tag:** phase-2 Nov 2026

## Team Members
* Elyas Hindi (ID: 182439) - Engineering Lead
* Omar Abu Khadair (ID: 162858) - Data Lead

## Project Overview
This repository contains the full Phase 2 Training Pipeline for the BuildMate AI model (based on GPT-2 124M), tailored for JUST academic rules and PC hardware compatibility. 

## Features Implemented
* **Continued Pretraining (CPT):** Fixed block size of 512 tokens using `train_cpt.py` with early stopping applied on the validation set.
* **Supervised Fine-Tuning (SFT):** Implemented via `train_sft.py` utilizing the `labels = -100` cross-entropy loss masking technique. Gradient updates are focused strictly on the answer tokens generated after the `<|assistant|>` special token.
* **Evaluation Harness:** Automated generation testing using `score.py` comparing Base, CPT, and SFT model outputs via greedy decoding.

## Hardware Used
* Local Environment (Windows 11 / Anaconda)
* NVIDIA GeForce RTX 3060 12GB VRAM (CUDA 12.1)

## How to Run
1. Run `python train_cpt.py` to adapt the model to the JUST domain text.
2. Run `python train_sft.py` to train the model on the Q&A dataset with loss masking.
3. Run `python score.py` to generate the evaluation CSV files in the `results/` folder.
