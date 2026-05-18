"""
End-to-end automation for ER-augmented MCTS comparison runs and baseline HPS sweeps.

For each requested environment the orchestrator:
  1. (optional) builds thts-run-toy-env via make
  2. runs the *tune* expr (ER tune or baselines HPS, depending on env config)
  3. (optional) runs a paired *baselines* expr -- same instance / num_trials /
     num_repeats -- if the env config defines `baseline_expr`
  4. picks the top-1 winner per algorithm via find_best_hyperparams.py
  5. writes a winners JSON next to the tune results
  6. (optional) invokes plot.py <plot_tag> --er-best-json=<path> to produce the
     comparison plot under plots/, if the env config defines `plot_tag`

Two flavors of env config:
  - ER-comparison envs (fl12 / s6 / tx5): full pipeline with paired baselines + plot.
  - HPS-only envs (fl16 / s10): only the HPS expr is run; winners JSON is written,
    no paired baselines, no plot.

Use --skip-tune / --skip-baselines / --plot-only to drive partial pipelines
when iterating on plotting after a long tune run has already finished.

Examples
--------
    # full pipeline for one env
    python run_experiment.py fl12

    # full pipeline for all three ER-comparison envs
    python run_experiment.py fl12 s6 tx5

    # baselines HPS only (no plot)
    python run_experiment.py fl16
    python run_experiment.py s10

    # tune already done, just rebuild the plot
    python run_experiment.py fl12 --plot-only

    # skip the build step (you already ran make)
    python run_experiment.py s6 --skip-build

The orchestrator does not babysit running the C++ binary -- the binary blocks
until it has run every RunID in the expr. If your tune expr takes hours/days,
launch this script under nohup / tmux / screen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Optional

# Each environment maps to its tune expr_id (required), an optional paired
# baseline expr_id, the results subdirectory, and an optional plot tag.
# - If `baseline_expr` is omitted, the baselines step is skipped.
# - If `plot_tag` is omitted, the plotting step is skipped.
# Add a new env here if you add an env to run_id.cpp.
ENV_CONFIGS: Dict[str, Dict[str, str]] = {
    "fl12": {
        "label":         "Frozen Lake 8x12",
        "tune_expr":     "063_fl12_er_tune",
        "baseline_expr": "064_fl12_baselines",
        "results_root":  "results/frozen_lake_env/FL_8x12",
        "plot_tag":      "compare_fl12_tune",
    },
    "s6": {
        "label":         "Sailing 6x6",
        "tune_expr":     "093_s6_er_tune",
        "baseline_expr": "094_s6_baselines",
        "results_root":  "results/sailing_env/6",
        "plot_tag":      "compare_s6_tune",
    },
    "tx5": {
        "label":         "Taxi 5x5",
        "tune_expr":     "103_tx5_er_tune",
        "baseline_expr": "104_tx5_baselines",
        "results_root":  "results/taxi_env/5",
        "plot_tag":      "compare_tx5_tune",
    },
    # HPS-only entries (single hyperparameter sweep, no paired baseline, no plot).
    # `<env>`     -> non-ER baselines HPS
    # `<env>_er`  -> ER-augmented variants HPS (same env / instance)
    "fl16": {
        "label":         "Frozen Lake 8x16 (baselines HPS)",
        "tune_expr":     "071_fl16_hps",
        "results_root":  "results/frozen_lake_env/FL_8x16_test",
    },
    "fl16_er": {
        "label":         "Frozen Lake 8x16 (ER HPS)",
        "tune_expr":     "073_fl16_er_tune",
        "results_root":  "results/frozen_lake_env/FL_8x16_test",
    },
    "s10": {
        "label":         "Sailing 10x10 (baselines HPS)",
        "tune_expr":     "121_s10_hps",
        "results_root":  "results/sailing_env/10",
    },
    "s10_er": {
        "label":         "Sailing 10x10 (ER HPS)",
        "tune_expr":     "123_s10_er_tune",
        "results_root":  "results/sailing_env/10",
    },
}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _print_header(text: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{text}\n{bar}", flush=True)


def _run(cmd, cwd: Optional[str] = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def build() -> None:
    _print_header("BUILD: make thts-run-toy-env")
    _run(["make", "-j4", "thts-run-toy-env"])


def run_expr(expr_id: str) -> None:
    binary = os.path.join(REPO_ROOT, "thts-run-toy-env")
    if not os.path.exists(binary):
        sys.exit(f"binary not found: {binary} (did you run make?)")
    _print_header(f"RUN: ./thts-run-toy-env {expr_id}")
    _run([binary, expr_id])


def find_winners(tune_dir: str, top_k: int) -> Dict[str, dict]:
    """
    Calls find_best_hyperparams.find_best_hyperparams in-process. Returns the
    same shape as that function (algorithm -> {best, candidates, num_total_configs}).
    """
    if not os.path.isdir(tune_dir):
        sys.exit(f"tune results not found: {tune_dir}")
    sys.path.insert(0, REPO_ROOT)
    try:
        from find_best_hyperparams import find_best_hyperparams as _fbh
    finally:
        sys.path.pop(0)
    return _fbh(tune_dir, top_k)


def write_winners_json(env_key: str, tune_dir: str, results: Dict[str, dict]) -> str:
    path = os.path.join(REPO_ROOT, tune_dir, "best_er_params.json")
    blob = {tune_dir: results}
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nWrote {path}", flush=True)
    return path


def plot(env_key: str, json_path: str) -> None:
    cfg = ENV_CONFIGS[env_key]
    _print_header(f"PLOT: {cfg['plot_tag']} (env: {cfg['label']})")
    _run([sys.executable, "plot.py", cfg["plot_tag"], f"--er-best-json={json_path}"])


def process_env(env_key: str, args: argparse.Namespace) -> None:
    cfg = ENV_CONFIGS[env_key]
    print(f"\n>>> {cfg['label']} <<<", flush=True)

    baseline_expr = cfg.get("baseline_expr")
    plot_tag = cfg.get("plot_tag")

    if not args.plot_only:
        if not args.skip_tune:
            run_expr(cfg["tune_expr"])
        if baseline_expr and not args.skip_baselines:
            run_expr(baseline_expr)

    tune_dir = os.path.join(cfg["results_root"], cfg["tune_expr"])
    _print_header(f"FIND WINNERS: {tune_dir}")
    results = find_winners(tune_dir, top_k=args.top_k)
    if not results:
        print(f"No winners found for {env_key}; skipping plot.", flush=True)
        return

    json_path = write_winners_json(env_key, tune_dir, results)

    if plot_tag and not args.skip_plot:
        plot(env_key, json_path)
    elif not plot_tag:
        print(f"No plot_tag configured for {env_key}; skipping plot.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "envs",
        nargs="*",
        default=list(ENV_CONFIGS.keys()),
        choices=list(ENV_CONFIGS.keys()),
        help="One or more of: fl12, s6, tx5 (default: all three).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `make thts-run-toy-env`. Use this if you've already built.",
    )
    parser.add_argument(
        "--skip-tune",
        action="store_true",
        help="Skip the *_er_tune run; assumes results already exist on disk.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the *_baselines run; assumes results already exist on disk.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip the plotting step. Useful when you only want to refresh the JSON.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Equivalent to --skip-tune --skip-baselines --skip-build.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K candidates to record per algorithm in the JSON / stdout (default: 5).",
    )
    args = parser.parse_args()

    if args.plot_only:
        args.skip_build = True
        args.skip_tune = True
        args.skip_baselines = True

    if not args.skip_build:
        build()

    for env_key in args.envs:
        process_env(env_key, args)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
