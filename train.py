import os
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    default_data_collator, Trainer, TrainingArguments, set_seed
)
from huggingface_hub import login, create_repo, upload_folder
import config
from data_processing import get_clean_dataset, get_preprocess_fn

def set_all_seeds():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    set_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def main():
    if not config.HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable before running.")
    
    os.environ["HF_TOKEN"] = config.HF_TOKEN
    login(token=config.HF_TOKEN)

    # 1. SETUP REPO
    try:
        print(f"Ensuring repository exists: {config.REPO_ID}")
        create_repo(repo_id=config.REPO_ID, token=config.HF_TOKEN, repo_type="model", exist_ok=True)
        print("✅ Repo ready.")
    except Exception as e:
        print(f"Note on repo creation: {e}")

    # 2. SET SEEDS
    set_all_seeds()

    # 3. GET AND CLEAN DATA
    dataset = get_clean_dataset(config.HF_TOKEN)

    # 4. LOAD MODEL & TOKENIZER
    print(f"Loading Model: {config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID,
        trust_remote_code=True,
        token=config.HF_TOKEN,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=config.HF_TOKEN,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. PREPROCESS DATASET
    preprocess_fn = get_preprocess_fn(tokenizer, config.MAX_SEQ_LENGTH)
    tokenized_train = dataset.map(preprocess_fn, batched=True)

    # 6. MASKING VERIFICATION
    print("\n" + "="*70)
    print("MULTI-SAMPLE MASKING VERIFICATION")
    print("="*70)

    for i in range(min(5, len(tokenized_train))):
        sample_input = tokenized_train[i]['input_ids']
        sample_label = tokenized_train[i]['labels']

        full_text = tokenizer.decode(
            [t for t in sample_input if t != tokenizer.pad_token_id],
            skip_special_tokens=False,
        )
        active_tokens = [t for t in sample_label if t != -100]
        learned_text  = tokenizer.decode(active_tokens, skip_special_tokens=False)

        print(f"\n[SAMPLE {i}]")
        print(f"--- FULL CONTEXT ---")
        print(repr(full_text))
        print(f"--- LEARNED TARGET ---")
        print(repr(learned_text))

        has_think  = "<think>" in learned_text
        has_header = (
            learned_text.strip().startswith("English:")
            or "assistant" in learned_text.lower()
        )

        if has_think:
            print(">> [!] WARNING: <think> tokens found in learned target — reasoning masking failed!")
        elif has_header:
            print(">> [!] WARNING: Template headers detected in learned target.")
        elif len(active_tokens) == 0:
            print(">> [!] WARNING: No tokens are being learned — check prompt_len calculation!")
        else:
            print(">> [SUCCESS]: Target starts cleanly at the translation.")
        print("-" * 40)
    print("="*70 + "\n")

    # 7. TRAINING
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        save_strategy="no",
        seed=config.SEED,
        learning_rate=config.LR,
        per_device_train_batch_size=config.TRAIN_BATCH,
        per_device_eval_batch_size=config.VAL_BATCH,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        weight_decay=config.WEIGHT_DECAY,
        num_train_epochs=config.NUM_EPOCHS,
        fp16=False,
        bf16=True,
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=default_data_collator,
    )

    print("Starting Training...")
    trainer.train()
    
    # 8. SAVE & UPLOAD
    print("Saving Model...")
    trainer.save_model(config.FULL_MODEL_PATH)
    tokenizer.save_pretrained(config.FULL_MODEL_PATH)  
    
    print(f"Uploading Model to {config.REPO_ID}...")
    upload_folder(
        folder_path=config.FULL_MODEL_PATH,
        repo_id=config.REPO_ID,
        repo_type="model",
        token=config.HF_TOKEN,
    )
    print(f"✅ Model successfully uploaded to https://huggingface.co/{config.REPO_ID}")

if __name__ == "__main__":
    main()