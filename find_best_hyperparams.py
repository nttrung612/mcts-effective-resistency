import os
import glob
import argparse
from collections import namedtuple

import pandas as pd


Score = namedtuple("Score", ["mean", "spread", "num_reps"])


def evaluate_csv(filepath):
    """
    Reads an eval_*.csv file produced by run_toy.cpp and returns:
      param_dict: dict of hyperparameter name -> value (strings)
      score: Score(mean, spread, num_reps) at the maximum num_trials, where
             - mean is the average mc_eval_mean across replicates,
             - spread is the (population) std of mc_eval_mean across replicates,
             - num_reps is the number of replicate rows that reached max_trials.
             Returns None for score if the file is malformed / empty.
    """
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        if len(lines) < 5:
            return None, None

        param_names = lines[0].strip().split(",")
        param_vals = lines[1].strip().split(",")
        param_dict = dict(zip(param_names, param_vals))

        # The 4th line (index 3) is the header for the actual data.
        df = pd.read_csv(filepath, skiprows=3)

        if df.empty or "num_trials" not in df.columns or "mc_eval_mean" not in df.columns:
            return param_dict, None

        max_trials = df["num_trials"].max()
        df_max = df[df["num_trials"] == max_trials]
        if df_max.empty:
            return param_dict, None

        score = Score(
            mean=float(df_max["mc_eval_mean"].mean()),
            spread=float(df_max["mc_eval_mean"].std(ddof=0)) if len(df_max) > 1 else 0.0,
            num_reps=int(len(df_max)),
        )
        return param_dict, score

    except Exception as exc:
        print(f"Error reading {filepath}: {exc}")
        return None, None


def format_params(param_dict):
    return ", ".join(f"{k}={v}" for k, v in param_dict.items() if k != "alg")


def find_best_hyperparams(base_dir, top_k):
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    algorithms = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )

    print(f"\n{'=' * 80}")
    print(f"Best Hyperparameters Analysis for: {base_dir}")
    print(f"{'=' * 80}")

    for algo in algorithms:
        algo_dir = os.path.join(base_dir, algo)
        # eval_*.csv only -- skip log_*_NNN.csv which has a different schema
        csv_files = glob.glob(os.path.join(algo_dir, "eval_*.csv"))
        if not csv_files:
            continue

        candidates = []
        for csv_file in csv_files:
            params, score = evaluate_csv(csv_file)
            if params is None or score is None:
                continue
            candidates.append((score, params, csv_file))

        if not candidates:
            continue

        candidates.sort(key=lambda c: c[0].mean, reverse=True)
        keep = candidates[: max(1, top_k)]

        print(f"\nAlgorithm: \033[1m{algo.upper()}\033[0m  ({len(candidates)} configs)")
        for rank, (score, params, csv_file) in enumerate(keep, start=1):
            label = "Best" if rank == 1 else f"#{rank}"
            colour_open, colour_close = ("\033[92m", "\033[0m") if rank == 1 else ("", "")
            print(
                f"  {label}: mean={score.mean:.4f}  "
                f"spread={score.spread:.4f}  "
                f"reps={score.num_reps}"
            )
            print(f"    File:   {csv_file}")
            print(f"    Params: {colour_open}{format_params(params)}{colour_close}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal hyperparameters from experiment results."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more results directories, e.g. results/frozen_lake_env/FL_8x12/063_fl12_er_tune",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Print the top K candidates per algorithm (default: 5). Set to 1 for the old behaviour.",
    )
    args = parser.parse_args()

    for directory in args.directories:
        find_best_hyperparams(directory, args.top_k)
