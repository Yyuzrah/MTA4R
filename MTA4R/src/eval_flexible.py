# src/eval_flexible.py
import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoForCausalLM, GPTNeoXForCausalLM
from countdown_utils import metric_fn

# Set deterministic execution for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

def load_model_and_tokenizer(ckpt_path):
    # Attempt to read the training config to determine the architecture
    config_path = os.path.join(ckpt_path, "exp_config.json")
    pe_type = "ape"
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            exp_config = json.load(f)
            pe_type = exp_config.get("pe_type", "ape")
    else:
        # Fallback logic based on path naming
        if "rope" in ckpt_path: pe_type = "rope"
        elif "ape" in ckpt_path: pe_type = "ape"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    print(f"Loading model from {ckpt_path} | Type: {pe_type}")
    
    if pe_type == "rope":
        model = GPTNeoXForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
        if torch.cuda.get_device_capability()[0] >= 8:
            model.to("cuda") 
    else:
        model = GPTNeoForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
        model.to("cuda")
    
    # Explicitly synchronize pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    return model, tokenizer, pe_type

def evaluate_on_dataset(model, tokenizer, data_path, batch_size=32, debug=False):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    prompts = []
    for sample in data:
        # [CRITICAL FIX]: Add a newline character (\n) after 'Operations: []'
        # The training data contains a newline here, and APE models are highly sensitive to this.
        prompt = f"{tokenizer.bos_token}Current State: {sample['target']}:{sample['nums']}, Operations: []\n"
        prompts.append(prompt)
        
    results = []
    desc = f"Eval {os.path.basename(data_path).split('.')[0]}"
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
        batch_prompts = prompts[i : i + batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # Greedy decoding
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # [FORCE DEBUG OUTPUT]
        # Print the input vs. output of the first sample to inspect model behavior
        if debug and i == 0:
            print("="*40)
            print(f"DEBUG MODEL GENERATION ({os.path.basename(data_path)})")
            print("-" * 20)
            print(f"[PROMPT]: {batch_prompts[0]}")
            print("-" * 20)
            print(f"[OUTPUT]: {decoded[0]}")
            print("="*40)
        
        for text in decoded:
            score, status = metric_fn(text)
            is_correct = 1 if score > 0 else 0
            results.append(is_correct)
            
    return np.mean(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--model_id", type=str, default=None, help="Specific model folder name to eval")
    parser.add_argument("--data_dir", type=str, default="data/experiment_500k")
    parser.add_argument("--output_dir", type=str, default="analysis/individual")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    val_sets = {
        "bfs": os.path.join(args.data_dir, "val_bfs.json"),
        "dfs": os.path.join(args.data_dir, "val_dfs.json"),
        "mixed": os.path.join(args.data_dir, "val_mixed.json")
    }

    if args.model_id:
        models_to_eval = [(args.model_id, os.path.join(args.checkpoints_dir, args.model_id, "final_model"))]
    else:
        models_to_eval = []
        if os.path.exists(args.checkpoints_dir):
            for exp_name in sorted(os.listdir(args.checkpoints_dir)):
                final_model_path = os.path.join(args.checkpoints_dir, exp_name, "final_model")
                if os.path.exists(final_model_path):
                    models_to_eval.append((exp_name, final_model_path))

    for exp_name, model_path in models_to_eval:
        output_path = os.path.join(args.output_dir, f"{exp_name}.json")
        
        # Removed the "result exists" check to force rerun for APE, as previous results were incorrect
        if os.path.exists(output_path) and "ape" not in exp_name:
            print(f"Skipping {exp_name}, result exists.")
            continue
            
        print(f"\n>>> Evaluating {exp_name} <<<")
        try:
            model, tokenizer, pe_type = load_model_and_tokenizer(model_path)
            
            # Force batch size to 1 for APE models
            current_batch_size = 1 if pe_type == "ape" else 32
            print(f"Using Batch Size: {current_batch_size} (Reason: {pe_type})")

            exp_scores = {}
            first_dataset = True
            for tag, path in val_sets.items():
                if os.path.exists(path):
                    # Print debug info only during the first dataset evaluation for each model
                    acc = evaluate_on_dataset(model, tokenizer, path, batch_size=current_batch_size, debug=first_dataset)
                    exp_scores[tag] = acc
                    print(f"    {tag}: {acc:.4f}")
                    first_dataset = False
            
            with open(output_path, "w") as f:
                json.dump(exp_scores, f, indent=4)
                
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {exp_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()