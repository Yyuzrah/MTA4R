import os
import json
import subprocess
import copy
import argparse

# === Experiment Configuration Matrix ===
MODEL_CONFIGS = {
    'small_deep':      {'n_layer': 12, 'n_embd': 192,  'n_head': 4},
    'small_wide':      {'n_layer': 3,  'n_embd': 384,  'n_head': 4},
    'small_balanced':  {'n_layer': 6,  'n_embd': 256,  'n_head': 4},
    
    'medium_deep':     {'n_layer': 24, 'n_embd': 512,  'n_head': 8},
    'medium_wide':     {'n_layer': 6,  'n_embd': 1024, 'n_head': 16},
    'medium_balanced': {'n_layer': 12, 'n_embd': 768,  'n_head': 12},
    
    'large_deep':      {'n_layer': 48, 'n_embd': 768,  'n_head': 12},
    'large_wide':      {'n_layer': 12, 'n_embd': 1536, 'n_head': 24},
    'large_balanced':  {'n_layer': 24, 'n_embd': 1024, 'n_head': 16},
}

DATASETS = {
    'bfs': 'train_bfs.json',
    'dfs': 'train_dfs.json',
    'mixed': 'train_mixed.json'
}

PE_TYPES = ['ape', 'rope']

# Base training configuration (Target Effective Batch Size = 32)
BASE_TEMPLATE = {
    "seed": 42,
    "data_dir": "data/experiment_500k",
    "val_file": "val_bfs.json",
    # "num_train": 100000, # Removed, using full dataset for training
    "context_length": 1024,
    "batch_size": 32,      # Default Batch Size
    "eval_steps": 2000,
    "log_steps": 500,
    "gradient_accumulation_steps": 1, # Default gradient accumulation
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "lr": 3e-4,
    "save_total_limit": 1,
    "save_steps": 5000,
    "wandb": {
        "project": "SoS_Repro_500k_Full",
        "entity": "YOUR_ENTITY_HERE", # WARNING: Ensure this remains anonymized for public repo!
        "dir": "wandb_logs"
    }
}

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/{job_name}_%j.out
#SBATCH --error=logs/slurm/{job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=long

source $HOME/.bashrc 2>/dev/null || source /etc/profile.d/conda.sh 2>/dev/null
conda activate ARDSCoT 2>/dev/null

export WANDB_MODE=offline
export PYTHONPATH=$PYTHONPATH:.
# Random port to prevent conflicts during DDP initialization
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)

echo "Starting training for {job_name} on port $MASTER_PORT"
accelerate launch --num_processes 1 --main_process_port $MASTER_PORT src/train_flexible.py --config {config_path} --wandb
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency", type=str, default=None, help="Slurm job ID to depend on")
    # Add an argument to selectively rerun Medium and Large models
    parser.add_argument("--only_heavy", action="store_true", help="Only submit medium and large models")
    args = parser.parse_args()

    config_dir = "configs/ablation_full"
    sbatch_dir = "scripts/sbatch_full"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(sbatch_dir, exist_ok=True)
    
    count = 0
    
    for model_name, arch in MODEL_CONFIGS.items():
        # Skip small models if only heavy models are specified
        if args.only_heavy and "small" in model_name:
            continue

        for data_name, data_file in DATASETS.items():
            for pe_type in PE_TYPES:
                
                exp_id = f"{model_name}_{data_name}_{pe_type}"
                
                # Check if the training is already completed
                if os.path.exists(f"checkpoints/{exp_id}/final_model"):
                    print(f"Skipping {exp_id} (Completed)")
                    continue

                config = copy.deepcopy(BASE_TEMPLATE)
                config.update(arch)
                config["train_file"] = data_file
                config["pe_type"] = pe_type
                config["arch_type"] = "gptneox" if pe_type == "rope" else "gptneo"
                config["name"] = exp_id
                config["output_dir"] = f"checkpoints/{exp_id}"
                
                # [Modification]: Apply memory optimization for Medium and Large models
                # Target Effective Batch Size = 32
                if "large" in model_name or "medium" in model_name:
                    print(f"Applying memory optimization for {exp_id} (BS=4, GA=8, Checkpoint=True)")
                    config["batch_size"] = 4
                    config["gradient_accumulation_steps"] = 8
                    config["gradient_checkpointing"] = True
                else:
                    # Keep default settings for Small models
                    config["gradient_checkpointing"] = False
                
                cfg_path = os.path.abspath(os.path.join(config_dir, f"{exp_id}.json"))
                with open(cfg_path, "w") as f:
                    json.dump(config, f, indent=4)
                
                sbatch_content = SLURM_TEMPLATE.format(
                    job_name=exp_id,
                    config_path=cfg_path
                )
                
                sbatch_path = os.path.join(sbatch_dir, f"{exp_id}.sh")
                with open(sbatch_path, "w") as f:
                    f.write(sbatch_content)
                
                cmd = ["sbatch"]
                if args.dependency:
                    cmd.append(f"--dependency=afterok:{args.dependency}")
                
                cmd.append(sbatch_path)
                
                print(f"Submitting {exp_id}...")
                subprocess.run(cmd, check=True)
                count += 1
                
    print(f"Successfully submitted {count} training jobs.")

if __name__ == "__main__":
    main()