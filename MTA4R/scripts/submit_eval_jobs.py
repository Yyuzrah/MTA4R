# scripts/submit_eval_jobs.py
import os
import subprocess

# Slurm template: A single evaluation typically takes 1-2 hours. 
# Requesting 4 hours as a safe margin.
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_{model_id}
#SBATCH --output=logs/eval_slurm/{model_id}.out
#SBATCH --error=logs/eval_slurm/{model_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=long

source $HOME/.bashrc 2>/dev/null || source /etc/profile.d/conda.sh 2>/dev/null
conda activate ARDSCoT 2>/dev/null
export PYTHONPATH=$PYTHONPATH:.

python src/eval_flexible.py --model_id {model_id} --data_dir data/experiment_500k --output_dir analysis/individual
"""

def main():
    checkpoints_dir = "checkpoints"
    results_dir = "analysis/individual"
    sbatch_dir = "scripts/sbatch_eval"
    log_dir = "logs/eval_slurm"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(sbatch_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    models = sorted(os.listdir(checkpoints_dir))
    submitted_count = 0

    for model_id in models:
        model_path = os.path.join(checkpoints_dir, model_id, "final_model")
        result_path = os.path.join(results_dir, f"{model_id}.json")

        # 1. Check if the model training is complete
        if not os.path.exists(model_path):
            continue

        # 2. Check if the model has already been evaluated
        if os.path.exists(result_path):
            print(f"Skipping {model_id} (Already evaluated)")
            continue

        # 3. Generate the Sbatch script
        sbatch_content = SLURM_TEMPLATE.format(model_id=model_id)
        sbatch_path = os.path.join(sbatch_dir, f"eval_{model_id}.sh")
        
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)

        # 4. Submit the job
        print(f"Submitting eval for {model_id}...")
        subprocess.run(["sbatch", sbatch_path], check=True)
        submitted_count += 1

    print(f"\nTotal evaluation jobs submitted: {submitted_count}")

if __name__ == "__main__":
    main()