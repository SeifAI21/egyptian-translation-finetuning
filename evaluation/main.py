import pandas as pd
from huggingface_hub import login
import config
from inference import run_inference
from metrics import Evaluator

def main():
    # Login to HuggingFace
    login(token=config.HF_TOKEN)

    # Load Data
    print("Loading Flores dataset...")
    en = pd.read_csv(config.DATA_PATHS["source"], encoding="utf-8-sig")
    ar = pd.read_csv(config.DATA_PATHS["target"], encoding="utf-8-sig")
    
    df = pd.DataFrame({
        "source": en["sentence"],
        "target": ar["sentence"]
    })

    # Run Inference
    for model_id, tokenizer_id in zip(config.MODELS, config.TOKENIZERS):
        print(f"\nProcessing Model: {model_id}")
        
        # We print a small sample for debugging like in the original code
        print("\nSOURCE (Egyptian Arabic) preview:")
        print(repr(df["source"].iloc[0]))
        print("TARGET (English reference) preview:")
        print(repr(df["target"].iloc[0]))
        
        df = run_inference(
            model_id, 
            tokenizer_id, 
            df, 
            batch_size=config.GENERATION_KWARGS["batch_size"], 
            device=config.GENERATION_KWARGS["device"]
        )
        print(f"{model_id} generation done!")

    # Save Checkpoint
    df.to_csv(config.OUTPUT_CSV, encoding="utf-8-sig", index=False)
    print(f"\nSaved generations to {config.OUTPUT_CSV}")

    # Evaluation
    print("\n========== FINAL SCORES ==========")
    evaluator = Evaluator()
    
    for model_id in config.MODELS:
        scores_basic = evaluator.eval_bleu_chrf(model_id, df)
        scores_comet = evaluator.eval_comet(model_id, df)
        print(f"\nModel: {model_id}")
        print(f"Basic Metrics: {scores_basic}")
        print(f"COMET Score: {scores_comet}")

if __name__ == "__main__":
    main()