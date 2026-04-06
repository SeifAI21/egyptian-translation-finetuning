import os

# IMPORTANT: Do not hardcode your token!
# Set it in your terminal before running: export HF_TOKEN="your_token_here"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ==========================================
# MODEL & TOKENIZER
# ==========================================
MODEL_ID = "Qwen/Qwen3-4B"
TOKENIZER_ID = "Qwen/Qwen3-4B"
REPO_ID = "SeifAI/Qwen3_4B_FT_SENTENCE_LEVEL_English_to_Egyptian_Arabic_EXP2"
FULL_MODEL_PATH = "./full_model_qwen3_4b_sentence"
OUTPUT_DIR = "./outputs_qwen3_4b_sentence"

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
LR = 3e-5
TRAIN_BATCH = 8
VAL_BATCH = 4
GRAD_ACCUM = 8
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
SAVE_STEPS = 500
EVAL_STEPS = 1000
LOGGING_STEPS = 100

MAX_SEQ_LENGTH = 256
SEED = 42

# ==========================================
# DATASET
# ==========================================
HF_DATASET_REPO_ID = "RanaGaber/SentenceLevelMT"
MIX_WITH_ORIGINAL_DATA = False
LOCAL_SAMPLES = None  # No Limit
HF_SAMPLES = None     # No Limit
