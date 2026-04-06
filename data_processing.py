import pandas as pd
from datasets import load_from_disk, Dataset, snapshot_download
import config

def get_clean_dataset(token):
    print(f"Downloading dataset from Hugging Face: {config.HF_DATASET_REPO_ID}")
    dataset_path = snapshot_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=token,
    )

    dataset_obj = load_from_disk(dataset_path)

    if isinstance(dataset_obj, dict) or hasattr(dataset_obj, "keys"):
        if "train" in dataset_obj:
            dataset = dataset_obj["train"]
        else:
            first_split = list(dataset_obj.keys())[0]
            dataset = dataset_obj[first_split]
            print(f"Using split: {first_split}")
    else:
        dataset = dataset_obj

    df = dataset.to_pandas()

    if 'arabic_sentence' in df.columns and 'english_sentence' in df.columns:
        df = df.rename(columns={'arabic_sentence': 'Egyptian', 'english_sentence': 'English'})
    elif 'arabic' in df.columns and 'english' in df.columns:
        df = df.rename(columns={'arabic': 'Egyptian', 'english': 'English'})

    if 'Egyptian' not in df.columns or 'English' not in df.columns:
        raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")

    df = df[['Egyptian', 'English']]

    # CRITICAL DATA CLEANING
    df = df.dropna(subset=["Egyptian", "English"])
    df = df[
        (df["Egyptian"].astype(str).str.strip() != "") &
        (df["English"].astype(str).str.strip()  != "")
    ]
    print(f"✅ Total Clean Rows for Training: {len(df)}")

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.shuffle(seed=config.SEED)
    return dataset

def _match_token_seq(ids, pos, pattern):
    """Return True if `pattern` starts exactly at `pos` inside `ids`."""
    return ids[pos : pos + len(pattern)] == pattern

def get_preprocess_fn(tokenizer, max_length):
    THINK_OPEN_IDS  = tokenizer.encode("<think>",  add_special_tokens=False)
    THINK_CLOSE_IDS = tokenizer.encode("</think>", add_special_tokens=False)

    def preprocess(examples):
        input_ids_list      = []
        labels_list         = []
        attention_mask_list = []

        for source, target in zip(examples["English"], examples["Egyptian"]):
            source = str(source) if source is not None else ""
            target = str(target) if target is not None else ""

            messages = [
                {"role": "user", "content": f"You are a translation expert. Translate: {source}\nEgyptian Arabic:"},
                {"role": "assistant", "content": target}
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            full_enc = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_ids = full_enc["input_ids"]

            prompt_only_messages = [
                {"role": "user", "content": f"You are a translation expert. Translate: {source}\nEgyptian Arabic:"},
            ]
            prompt_only = tokenizer.apply_chat_template(
                prompt_only_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_only_enc = tokenizer(
                prompt_only,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            prompt_len = len(prompt_only_enc["input_ids"])

            labels     = [-100] * len(full_ids)
            i          = prompt_len
            in_thinking = False

            while i < len(full_ids):
                if _match_token_seq(full_ids, i, THINK_OPEN_IDS):
                    in_thinking = True
                    i += len(THINK_OPEN_IDS)
                    continue

                if in_thinking and _match_token_seq(full_ids, i, THINK_CLOSE_IDS):
                    in_thinking = False
                    i += len(THINK_CLOSE_IDS)
                    continue

                if not in_thinking:
                    labels[i] = full_ids[i]

                i += 1

            padding_length = max_length - len(full_ids)
            input_ids      = full_ids + [tokenizer.pad_token_id] * padding_length
            labels         = labels   + [-100]                   * padding_length
            attention_mask = [1] * len(full_ids) + [0]          * padding_length

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        return {
            "input_ids":       input_ids_list,
            "labels":          labels_list,
            "attention_mask":  attention_mask_list,
        }
    return preprocess