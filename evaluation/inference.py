import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

def clean_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(
        r"^(The following is a translation.*?:)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()

def run_inference(model_id, tokenizer_id, df, batch_size=128, device="cuda", debug=True):
    print(f"\n========== LOADING {model_id} ==========")
    sources = df["source"].tolist()
    generated_text = []

    if "Nile-Chat" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"  

        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            torch_dtype="auto",   
            device_map="auto",    
        )

        for i in tqdm(range(0, len(sources), batch_size)):
            batch = sources[i:i + batch_size]
            batch_messages = [
                [{
                    "role": "user",
                    "content":
                        "You are a translation expert. Translate the following English sentences to Egyptian Arabic.\n"
                        f"English: {s}\n"
                        "Egyptian Arabic:"
                }] for s in batch
            ]
            outputs = pipe(
                batch_messages, 
                batch_size=len(batch),
                max_new_tokens=256, 
                do_sample=False
            )
            for out in outputs:
                assistant_response = out[0]["generated_text"][-1]["content"].strip()
                generated_text.append(clean_output(assistant_response))
            
            if debug and i == 0:
                print(f"\n===== SAMPLE (NILE) =====\n{repr(generated_text[0])}")
        
        del pipe
        del tokenizer
        torch.cuda.empty_cache()

    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()

        for i in tqdm(range(0, len(sources), batch_size)):
            batch = sources[i:i + batch_size]
            batch_messages = [
                [{
                    "role": "user",
                    "content":
                        "You are a translation expert. Translate the following English sentences to Egyptian Arabic.\n"
                        f"English: {s}\n"
                        "Egyptian Arabic:"
                }] for s in batch
            ]
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_generation_prompt=True,
                return_dict=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[:, input_len:]
            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            for t in translations:
                generated_text.append(clean_output(t))
                
            if debug and i == 0:
                print(f"\n===== SAMPLE (QWEN) =====\n{repr(generated_text[0])}")

        del model
        del tokenizer
        torch.cuda.empty_cache()

    col_name = model_id.split("/")[-1]
    df[col_name] = generated_text
    return df