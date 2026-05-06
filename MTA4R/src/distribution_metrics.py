# src/distribution_metrics.py
import os
import json
import re
import numpy as np
import tqdm
from scipy.spatial.distance import cdist
import ot  # Requires installation: pip install POT

def extract_structural_features(trajectory_text):
    """
    Rigorous feature extraction specific to the SoS format.
    Example input format: "Generated Node #0,1: 47:[61, 24] Operation: 66-42=24\n..."
    """
    features = np.zeros(12, dtype=np.float32)
    
    if not isinstance(trajectory_text, str) or not trajectory_text:
        return features

    # 1. Extract Node IDs (e.g., "0,1,2")
    # The original SoS code output format typically contains "Node #0,1" or "Node #0,1,2"
    node_matches = re.findall(r"Node #(\d+(?:,\d+)*)", trajectory_text)
    
    # Basic statistics: number of operators (reflects heuristic preference)
    features[10] = trajectory_text.count('*') + trajectory_text.count('/')
    features[11] = trajectory_text.count('+') + trajectory_text.count('-')
    features[9] = 1.0 if "Goal Reached" in trajectory_text else 0.0

    if not node_matches:
        return features
    
    # 2. Topological features
    # The number of commas in the ID + 1 equals the depth (e.g., "0" -> depth 1, "0,1" -> depth 2)
    depths = [s.count(',') + 1 for s in node_matches]
    node_paths = [tuple(map(int, s.split(','))) for s in node_matches]

    features[0] = max(depths)       # Max Depth
    features[1] = np.mean(depths)   # Mean Depth
    features[2] = np.std(depths)    # Std Depth
    features[3] = len(node_matches) # Total Nodes Visited

    # 3. Dynamic features (Core difference between BFS and DFS)
    moves_down = 0
    moves_up = 0
    moves_side = 0
    
    if len(node_paths) > 1:
        for i in range(1, len(node_paths)):
            len_prev = len(node_paths[i-1])
            len_curr = len(node_paths[i])
            
            if len_curr > len_prev:
                moves_down += 1
            elif len_curr < len_prev:
                moves_up += 1
            else:
                # Depth unchanged, but moving, usually intra-layer traversal in BFS
                moves_side += 1 

        total_moves = len(node_paths) - 1
        if total_moves > 0:
            features[4] = moves_down / total_moves
            features[5] = moves_up / total_moves
            features[6] = moves_side / total_moves
    
    # 4. Pruning features
    prunes = len(re.findall(r"unequal: No Solution", trajectory_text))
    features[7] = prunes
    
    # Calculate the number of Generated Nodes to use as the denominator
    gen_nodes_count = len(re.findall(r"Generated Node #", trajectory_text))
    if gen_nodes_count > 0:
        features[8] = prunes / gen_nodes_count
        
    return features

def load_and_featurize(data_path, label, max_process=None):
    """Load JSON data and extract features."""
    print(f"[{label}] Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist.")

    data = []
    with open(data_path, 'r') as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
        except json.JSONDecodeError:
            # Handle JSON Lines if necessary
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    if max_process is not None and len(data) > max_process:
        # Even if the dataset is large, we only process a subset of features for Metrics 
        # calculation to save time. For a 500k dataset, it is recommended to extract 
        # at least 50k to adequately represent the distribution.
        import random
        random.seed(42)
        data = random.sample(data, max_process)
    
    print(f"[{label}] Extracting features from {len(data)} samples...")
    feature_vectors = []
    for item in tqdm.tqdm(data, desc=f"Featurizing {label}"):
        # SoS dataset key is usually "search_path"
        if "search_path" in item:
            text = item["search_path"]
            feature_vectors.append(extract_structural_features(text))
    
    return np.array(feature_vectors)

def compute_metrics(bfs_path, dfs_path, mixed_path, sample_size=2000):
    """
    Compute distribution distances between the three datasets.
    Note: To compute Wasserstein distance, we need to downsample (sample_size),
    otherwise the O(N^2) complexity is unfeasible on 500k data.
    """
    # Extract features (Use a larger sample size for feature extraction, e.g., 20000)
    feat_bfs = load_and_featurize(bfs_path, "BFS", max_process=20000)
    feat_dfs = load_and_featurize(dfs_path, "DFS", max_process=20000)
    feat_mixed = load_and_featurize(mixed_path, "MIXED", max_process=20000)

    # Normalization (Z-Score) - Using joint distribution parameters of all three datasets
    valid_vecs = [v for v in [feat_bfs, feat_dfs, feat_mixed] if len(v) > 0]
    if not valid_vecs:
        return "No valid data found."
        
    all_vecs = np.vstack(valid_vecs)
    mean = np.mean(all_vecs, axis=0)
    std = np.std(all_vecs, axis=0)
    std[std == 0] = 1.0 # Avoid division by zero

    norm_bfs = (feat_bfs - mean) / std
    norm_dfs = (feat_dfs - mean) / std
    norm_mixed = (feat_mixed - mean) / std

    # Define a helper function for distance calculation
    def calc_w_dist(v1, v2, n_samples):
        if len(v1) == 0 or len(v2) == 0: return float('nan')
        # Randomly sample n_samples for distance matrix calculation
        idx1 = np.random.choice(len(v1), min(len(v1), n_samples), replace=False)
        idx2 = np.random.choice(len(v2), min(len(v2), n_samples), replace=False)
        s1 = v1[idx1]
        s2 = v2[idx2]
        
        M = cdist(s1, s2, 'euclidean')
        M /= (np.max(M) + 1e-9) # Normalize the distance matrix
        
        p = np.ones(len(s1)) / len(s1)
        q = np.ones(len(s2)) / len(s2)
        
        return ot.sinkhorn2(p, q, M, 0.1) # Sinkhorn Regularization

    print("\n=== Distribution Metrics Report ===")
    w_bfs_dfs = calc_w_dist(norm_bfs, norm_dfs, sample_size)
    w_bfs_mix = calc_w_dist(norm_bfs, norm_mixed, sample_size)
    w_dfs_mix = calc_w_dist(norm_dfs, norm_mixed, sample_size)
    
    results = {
        "W_dist_BFS_DFS": w_bfs_dfs,
        "W_dist_BFS_MIXED": w_bfs_mix,
        "W_dist_DFS_MIXED": w_dfs_mix,
        "Ratio_Mixed_Pos": w_bfs_mix / (w_bfs_dfs + 1e-9) # Theoretically should be close to 0.5 if Mixed is a linear interpolation
    }
    
    print(json.dumps(results, indent=4))
    return results

if __name__ == "__main__":
    # Test entry point
    pass