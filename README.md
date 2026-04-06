# Egyptian Arabic Translation Fine-Tuning

This repository contains the fine-tuning pipeline for language models (like Qwen3, Nile-Chat, Gemma, etc.) to translate English to Egyptian Arabic. It supports standard Supervised Fine-Tuning (SFT) masking as well as specialized reasoning-token masking for models that use `<think>` blocks.

## Datasets
The pipeline can be configured to train on either Sentence-Level or Document-Level datasets.

* **Sentence-Level Dataset:** [RanaGaber/SentenceLevelMT](https://huggingface.co/datasets/RanaGaber/SentenceLevelMT)
* **Document-Level Dataset:** [SeifAI/FineTranselation_EGY_filtered_docs](https://huggingface.co/datasets/SeifAI/FineTranselation_EGY_filtered_docs)

## Features
* **Universal Masking Toggle:** Set `MASK_REASONING_TOKENS = True` in `config.py` for models like Qwen3 that generate `<think>` blocks to mask the reasoning appropriately. Set to `False` for standard models like Nile-Chat or Gemma to use standard SFT masking.
* **Memory Efficient:** Uses flash attention, gradient checkpointing, and bf16 mixed-precision training.
* **Hugging Face Hub Integration:** Automatically fetches datasets and pushes models directly to the Hugging Face Hub.

## Setup & Execution
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Export your Hugging Face Token:
```bash
export HF_TOKEN="your_hf_token_here"
```

3. Update `config.py` with your preferred Target Model, Dataset, and output repositories.

4. Run the training:
```bash
python train.py
```