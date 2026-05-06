# src/generate_dataset_SOS.py
import argparse
import random
import json
import os
import tqdm
import sys

# Import base project modules
from countdown import CountDown
from countdown_utils import sum_heuristic, mult_heuristic, simple_rating
from countdown_bfs import bfs
from countdown_dfs import dfs
# Import custom metrics module
from distribution_metrics import compute_metrics

def generate_data(args, search_type, output_filename):
    print(f"\nGenerating {args.num_samples} samples for strategy: {search_type}...")
    
    # Initialize the target list with a fixed seed for reproducibility
    random.seed(args.seed)
    target_nums = [i for i in range(10, args.max_target+1)]
    # Note: To maximize data diversity, targets are randomly sampled within the loop
    
    data_samples = []
    
    # Initialize progress bar
    pbar = tqdm.tqdm(total=args.num_samples)
    
    while len(data_samples) < args.num_samples:
        # Parameter setting logic (Referencing the original generate.py)
        start_size = random.randint(args.min_range, args.start_range)
        cd = CountDown(args.max_target, start_size)
        
        # Max Nodes estimation (Used for calculating the evaluation Rating)
        max_rating = 1152 # Default for start_size=4
        if start_size == 3: max_rating = 48
        elif start_size == 5: max_rating = 46080
        
        target = random.choice(target_nums)
        try:
            nums, solution = cd.generate(target)
        except ValueError:
            continue # Skip invalid targets
            
        no_backtrack_trace = cd.convert_to_path(target, nums, solution)
        
        # Search strategy selection
        search_path = ""
        heuristic = None
        
        if search_type == "dfs":
            heuristic = sum_heuristic # Default configuration
            search_path = dfs(target, nums, heuristic=heuristic, threshold=target)
            heuristic_name = "sum_heuristic"
            actual_search_type = "dfs"
            
        elif search_type == "bfs":
            heuristic = mult_heuristic # Default configuration
            # Random selection of BFS beam_size as described in the original paper
            beam_size = random.choice([1, 2, 3, 4, 5]) 
            search_path = bfs(target, nums, beam_size, heuristic=heuristic)
            heuristic_name = "mult_heuristic"
            actual_search_type = f"bfs_{beam_size}"
            
        elif search_type == "mixed":
            # 50% probability for BFS, 50% probability for DFS
            if random.random() < 0.5:
                # DFS Logic
                heuristic = random.choice([sum_heuristic, mult_heuristic])
                search_path = dfs(target, nums, heuristic=heuristic, threshold=target)
                actual_search_type = "dfs"
            else:
                # BFS Logic
                heuristic = random.choice([sum_heuristic, mult_heuristic])
                beam_size = random.choice([1, 2, 3, 4, 5])
                search_path = bfs(target, nums, beam_size, heuristic=heuristic)
                actual_search_type = f"bfs_{beam_size}"
            heuristic_name = heuristic.__name__

        # Rating calculation
        if "Goal Reached" in search_path:
            rating = 1. - simple_rating(search_path) / max_rating
            rating = max(0., rating)
        else:
            rating = 0.

        sample = {
            "nums": nums,
            "target": target,
            "solution": solution,
            "search_path": search_path,
            "rating": rating,
            "search_type": actual_search_type,
            "optimal_path": no_backtrack_trace,
            "heuristic": heuristic_name
        }
        
        data_samples.append(sample)
        pbar.update(1)
        
    pbar.close()
    
    # Save generated data to file
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(data_samples, f, indent=4)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500000)
    parser.add_argument("--data_dir", type=str, default="data/experiment_500k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_target", type=int, default=100)
    parser.add_argument("--min_range", type=int, default=3)
    parser.add_argument("--start_range", type=int, default=4) # Restrict to 3-4; generating 5 is too computationally expensive
    args = parser.parse_args()

    # 1. Generate BFS dataset
    bfs_file = os.path.join(args.data_dir, "train_bfs.json")
    generate_data(args, "bfs", bfs_file)

    # 2. Generate DFS dataset
    dfs_file = os.path.join(args.data_dir, "train_dfs.json")
    generate_data(args, "dfs", dfs_file)

    # 3. Generate Mixed dataset
    mixed_file = os.path.join(args.data_dir, "train_mixed.json")
    generate_data(args, "mixed", mixed_file)
    
    # 4. Generate validation sets (Small scale, for evaluation purposes)
    val_args = argparse.Namespace(**vars(args))
    val_args.num_samples = 2000
    generate_data(val_args, "bfs", os.path.join(args.data_dir, "val_bfs.json"))
    generate_data(val_args, "dfs", os.path.join(args.data_dir, "val_dfs.json"))

    # 5. Compute distribution metrics on the generated dataset
    print("\nComputing Distribution Metrics on Generated Data...")
    metrics = compute_metrics(bfs_file, dfs_file, mixed_file)
    
    # Save the computed metrics results
    with open(os.path.join(args.data_dir, "distribution_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)