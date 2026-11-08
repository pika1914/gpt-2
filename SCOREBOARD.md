# Phase 2 Scoreboard

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Base model | High | ~5% | 0% | GPT-2 small (124M, gpt2) |
| + CPT | Improved | ~15% | 0% | 3 passes, lr=5e-5, block_size=512 |
| + SFT | Low | ~85% | 100% | Answer-only loss masking (-100) applied |

**Error Analysis:**
The SFT model successfully learned the turn-taking `<|user|>` and `<|assistant|>` structure. Loss masking (`labels = -100`) prevented the model from echoing user prompts and forced it to memorize direct responses. Primary residual failures on a 124M parameter model include minor hardware hallucination due to limited parametric memory, requiring a RAG pipeline in Phase 3.
