import os
import glob
import pandas as pd
import argparse
from collections import defaultdict

def evaluate_csv(filepath):
    """
    Reads the custom formatted CSV file and returns:
    - param_dict: dictionary of hyperparameters
    - mean_val_at_max_trials: the mean mc_eval_mean at the maximum num_trials across replicates
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 5:
            return None, None
            
        param_names = lines[0].strip().split(',')
        param_vals = lines[1].strip().split(',')
        param_dict = dict(zip(param_names, param_vals))
        
        # Read the rest of the file into a pandas dataframe
        # The 4th line (index 3) is the header for the actual data
        df = pd.read_csv(filepath, skiprows=3)
        
        if df.empty or 'num_trials' not in df.columns or 'mc_eval_mean' not in df.columns:
            return param_dict, None
            
        max_trials = df['num_trials'].max()
        df_max_trials = df[df['num_trials'] == max_trials]
        
        mean_val = df_max_trials['mc_eval_mean'].mean()
        return param_dict, mean_val
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def find_best_hyperparams(base_dir):
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    algorithms = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"\n{'='*80}")
    print(f"Best Hyperparameters Analysis for: {base_dir}")
    print(f"{'='*80}")
    
    for algo in algorithms:
        algo_dir = os.path.join(base_dir, algo)
        csv_files = glob.glob(os.path.join(algo_dir, "*.csv"))
        
        if not csv_files:
            continue
            
        best_val = -float('inf')
        best_file = None
        best_params = None
        
        for csv_file in csv_files:
            params, val = evaluate_csv(csv_file)
            if val is not None and val > best_val:
                best_val = val
                best_file = csv_file
                best_params = params
                
        if best_file is not None:
            print(f"\nAlgorithm: \033[1m{algo.upper()}\033[0m")
            print(f"  Best Value (mean @ max trials): {best_val:.4f}")
            print(f"  Best File: {best_file}")
            
            param_str = ", ".join([f"{k}={v}" for k, v in best_params.items() if k != 'alg'])
            print(f"  Optimal Hyperparams: \033[92m{param_str}\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal hyperparameters from experiment results.")
    parser.add_argument("directories", nargs="+", help="Directories containing experiment results (e.g., results/frozen_lake_env/FL_8x12_test/052_fl12_test)")
    args = parser.parse_args()
    
    for directory in args.directories:
        find_best_hyperparams(directory)
