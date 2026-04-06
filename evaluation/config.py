# Configuration for evaluation
import os

# IMPORTANT: Never hardcode your token in GitHub!
# Set it in your terminal before running: export HF_TOKEN="your_token_here"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

DATA_PATHS = {
    "source": "./data/flores_eng_devtest.csv",
    "target": "./data/flores_egy_devtest.csv"
}

OUTPUT_CSV = "./experiment1_decodermodels_flores.csv"

# Models to test
MODELS = [
    "SeifAI/Gemma3_1B_IT_FT_SENTENCE_LEVEL_English_to_Egyptian_Arabic",
    "RanaGaber/gemma_4B_FT_en-eg",
    "SeifAI/Nile_Chat_FT_4B_SENTENCE_LEVEL_English_to_Arabic",
    "SeifAI/Qwen3_4B_FT_SENTENCE_LEVEL_English_to_Egyptian_Arabic",
    "SeifAI/Qwen3_1_7B_FT_SENTENCE_LEVEL_English_to_Egyptian_Arabic",
]

# Tokenizers associated with the models
TOKENIZERS = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "MBZUAI-Paris/Nile-Chat-4B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B"
]

GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "batch_size": 128,
    "device": "cuda",
    "do_sample": False
}