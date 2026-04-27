# Experiment Guide — Tuning ER algorithms & running baselines

This guide tells you, end to end, how to tune the hyperparameters of the
ER-augmented algorithms (ER-UCT, ER-Fixed-Depth-UCT, ER-MENTS, ER-RENTS,
ER-TENTS) and compare them against the baseline methods that have already
been tuned upstream. Three environments are supported: Frozen Lake 8×12,
Sailing 6×6, and Taxi 5×5.

> If you have not built the binary yet, run `make thts-run-toy-env` from the
> repo root. (You can ignore the `googletest` step in `README.md`; tests are
> not used here.)

---

## 1. Concepts

Every "experiment" is identified by an `<expr_id>` string and dispatches in
`src/toy_envs/run_id.cpp` to a list of `RunID`s. Each `RunID` is one
(environment, algorithm, hyperparameters, num_repeats, ...) tuple. Running

```
./thts-run-toy-env <expr_id>
```

executes every `RunID` in the list sequentially (one alg at a time), with
threading inside each run, and writes results to:

```
results/<env_id>/<env_instance_id>/<expr_id>/<alg_id>/eval_<params>.csv
results/<env_id>/<env_instance_id>/<expr_id>/<alg_id>/log_<params>_<replicate>.csv
results/<env_id>/<env_instance_id>/<expr_id>/<alg_id>/tree_<params>.txt
```

`eval_<params>.csv` is the file you score with `find_best_hyperparams.py` and
plot with `plot.py`. Header rows: `replicate,num_trials,mc_eval_mean,mc_eval_std`.

Hyperparameters are encoded both in the filename (`,`-joined `key=value`)
and in the first two lines of each CSV.

---

## 2. The expr_ids you care about

Six experiment IDs cover the workflow:

| Step | Frozen Lake 8×12 | Sailing 6×6 | Taxi 5×5 |
|---|---|---|---|
| **ER tune** (grid search ER algos) | `063_fl12_er_tune` | `093_s6_er_tune` | `103_tx5_er_tune` |
| **Baselines** (paired, untuned) | `064_fl12_baselines` | `094_s6_baselines` | `104_tx5_baselines` |

Each ER-tune expr is paired with a baseline expr that uses the **same**
training instance, `num_trials`, `num_repeats`, and `num_threads`, so the
two CSVs plot directly against each other on the same x-axis. Baselines use
hyperparameters that were already tuned in the previous (Dam et al.) work
and are hardcoded in `run_id.cpp`.

You can also re-use the older comparison runs if you want extra data points:

| | FL12 | S6 | TX5 |
|---|---|---|---|
| Pre-existing HPS (baselines + ER@c2=1) | `051_fl12_hps` | `091_s6_hps` | `101_tx5_hps` |
| Pre-existing test run (baselines + ER@c2=1) | `052_fl12_test` | `092_s6_test` | `102_tx5_test` |
| Combined snapshot on test instance | `062_fl12_requested_algs` | — | — |

These hardcode `er_c2 = 1.0` (and `power_mean_p = 2.0` on TX5) for ER
variants — they're useful only if you accept un-tuned ER. Once you've run
the `*_er_tune` expr, you should compare against the paired `*_baselines`
expr instead.

---

## 3. End-to-end workflow

The recipe below uses Frozen Lake as the running example. Sailing and Taxi
are identical, just swap the expr IDs.

### Step 1 — Build

```bash
make thts-run-toy-env
```

Re-run this after editing `run_id.cpp` (or any .cpp/.h under `src/` or
`include/`). The default `Makefile` already builds the toy-env binary.

### Step 2 — Run the ER tune

```bash
./thts-run-toy-env 063_fl12_er_tune
```

This sweeps the ER-augmented algorithms over the grid in
`run_id.cpp:FL12_063_ER_TUNE`. Output lands in
`results/frozen_lake_env/FL_8x12/063_fl12_er_tune/<alg_id>/`. (See the
"Hyperparameter grids" section for the exact ranges.)

This is by far the most expensive step. See "Compute cost" below before
launching.

### Step 3 — Pick best params per algorithm

```bash
python find_best_hyperparams.py results/frozen_lake_env/FL_8x12/063_fl12_er_tune
```

For each `<alg_id>` directory it walks the `eval_*.csv` files, takes the
`mc_eval_mean` averaged across the 5 replicates at the **final** `num_trials`,
and prints the (params, score) of the winner per algorithm. Save this output —
you'll either paste it into a follow-up expr_id (Step 5) or use it directly to
identify the right `eval_*.csv` for plotting.

### Step 4 — Run the baselines on the same training instance

```bash
./thts-run-toy-env 064_fl12_baselines
```

Output lands in `results/frozen_lake_env/FL_8x12/064_fl12_baselines/<alg_id>/`.
Baselines: UCT, Fixed-Depth-UCT, PUCT, MENTS, RENTS, TENTS, DENTS, DBMENTS, EST.
These use the previously-tuned hyperparameters (hardcoded), so this expr is
much cheaper than Step 2. `064` writes to the same `FL_8x12` instance
directory as `063_fl12_er_tune`, so both feed a single comparison plot
(Step 5).

### Step 5 — (Recommended) Test the tuned ER algorithms on the held-out instance

`*_er_tune` runs on the **train** instance for proper hyperparameter
selection, but your final comparison should be on the **test** instance. To
get that, add a small expr_id to `run_id.cpp` — call it `065_fl12_er_test`
— that runs each ER algorithm **once** on `FL_8x12_TEST` with the best
params from Step 3. Skeleton:

```cpp
// expr id: 065_fl12_er_test (add to run_id.h: FL12_065_ER_TEST)
if (expr_id == FL12_065_ER_TEST) {
    string env_id = FL_ENV_ID;
    string env_instance_id = FL_8x12_TEST;
    int num_trials = 150000;
    int max_trial_length = 100;
    int trials_log_delta = 250;
    int mc_eval_trials_delta = 250;
    int rollouts_per_mc_eval = 100;
    int num_repeats = 10;      // bump higher (e.g. 25) for paper-quality bands
    int num_threads = 16;
    int eval_threads = 16;

    // Paste the per-algorithm winners from find_best_hyperparams.py here:
    run_ids->push_back(RunID(env_id, env_instance_id, expr_id,
        ALG_ID_ER_UCT, {{PARAMS_ID_UCT_BIAS, /*winner*/},
                        {PARAMS_ID_UCT_ER_C2, /*winner*/},
                        {PARAMS_ID_UCT_POWER_MEAN_P, /*winner*/}},
        num_trials, max_trial_length, trials_log_delta,
        mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats,
        num_threads, eval_threads));
    // ... repeat for ER_FIXED_DEPTH_UCT, ER_MENTS, ER_RENTS, ER_TENTS ...
    return run_ids;
}
```

Add the matching constant in `include/toy_envs/run_id.h`:
```cpp
static const std::string FL12_065_ER_TEST = "065_fl12_er_test";
```

Then `make thts-run-toy-env && ./thts-run-toy-env 065_fl12_er_test`. Pair this
with a separate `066_fl12_baselines_test` (clone `064_fl12_baselines` but
change `env_instance_id` to `FL_8x12_TEST`) to get the apples-to-apples test
plot.

For Sailing and Taxi the same pattern applies; the `092_s6_test` /
`102_tx5_test` exprs already run baselines on the test instance, but they
hardcode untuned ER (`er_c2 = 1.0`). You'll want a similar `095_s6_er_test`
/ `105_tx5_er_test` per the skeleton above.

### Step 6 — Plot ER vs baselines

`plot.py` ships with three tag-driven comparison blocks at the bottom of
its `__main__`: `compare_fl12_tune`, `compare_s6_tune`, `compare_tx5_tune`.
Each one reads the best ER configurations from a small dict you fill in by
hand, plus all baseline `eval_*.csv`s globbed from the paired baselines
expr.

1. Open `plot.py` and find `FL12_ER_BEST = { ... }` (likewise `S6_ER_BEST`,
   `TX5_ER_BEST`). The dict starts out commented:
   ```python
   FL12_ER_BEST = {
       # "er-uct":             "eval_bias=...,er_c2=...,power_mean_p=....csv",
       # "er-fixed-depth-uct": "eval_bias=...,er_c2=...,power_mean_p=....csv",
       # "er-ments":           "eval_temp=...,epsilon=...,er_c2=....csv",
       # "er-rents":           "eval_temp=...,epsilon=...,er_c2=....csv",
       # "er-tents":           "eval_temp=...,epsilon=...,er_c2=....csv",
   }
   ```
2. Paste the basenames of the winning `eval_*.csv` files reported by
   `find_best_hyperparams.py`, uncommenting the lines as you go. The basename
   is just the file inside `<er_dir>/<alg>/`, not the full path. (For sailing,
   include `default_q_value=-200` in the filename, since the tune script
   passes it explicitly.)
3. Run:
   ```bash
   python plot.py compare_fl12_tune
   python plot.py compare_s6_tune
   python plot.py compare_tx5_tune
   ```
   Output PNGs land in `plots/compare_<env>_tune.png`.

The comparison blocks call `make_plot(..., er_vs_baseline_mode=True)`, which
groups each `(BASELINE, ER-BASELINE)` pair under one colour and draws the
baseline dashed and the ER variant solid -- so MENTS/ER-MENTS, RENTS/ER-RENTS,
TENTS/ER-TENTS, UCT/ER-UCT, FIXED-DEPTH-UCT/ER-FIXED-DEPTH-UCT each show up
as a clean dashed-vs-solid pair in the same hue. Algorithms without an ER
pair (PUCT, BTS, DENTS) stay solid in their own colour.

The shaded band around each line is **seaborn's default 95% bootstrap CI of
the cross-replicate mean** (5 replicates here -- the band is wide; bump
`num_repeats` in your `065_*_er_test` follow-up for tighter bands). The
single-snapshot `mc_eval_std` column from the CSV is intentionally not used:
what you want plotted is the variability *across independent MCTS runs*,
which is exactly what cross-replicate aggregation gives you.

If you want to plot something other than these three blocks, the existing
`if "<tag>" in sys.argv: ...` switchboard above still works -- pick an
existing tag like `000_fig_fl` for the older paper figures, or add your own
block following the same pattern.

---

## 4. Hyperparameter grids

What `*_er_tune` actually sweeps:

### ER-UCT, ER-Fixed-Depth-UCT (UCT family)

| Parameter | Values |
|---|---|
| `bias` (= C₁) | `USE_AUTO_BIAS, 0.1, 0.3, 1.0, 3.0, 10.0` (6) |
| `er_c2` (= C₂) | `0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0` (9) |
| `power_mean_p` | `1.0, 2.0, 4.0, ∞` (4) |

= **216 configs/alg × 2 algs = 432 RunIDs** per `*_er_tune` expr.

`USE_AUTO_BIAS` triggers PROST adaptive bias (sets bias to `max(|child.avg_return|)`
at each node, lower-bounded by 0.001). Other values use a fixed bias.
`power_mean_p = 1.0` reduces to arithmetic-mean backup; `∞` is the max
operator.

After our recent fix, the ER bonus `c2 / N(s,a)` is added **outside** the
`bias · prior` factor, so `bias` and `er_c2` are now genuinely independent
coefficients (this matches `ermcts.tex` Eq. (combined_bonus)). If you ran
this expr before the fix, your old results were biased — re-run.

### ER-MENTS, ER-RENTS, ER-TENTS (convex regularized family)

| Parameter | Values |
|---|---|
| `temp` (= τ) | `0.001, 0.01, 0.05, 0.1, 0.5` (5) |
| `epsilon` | `0.1, 0.3, 1.0, 2.0, 5.0` (5) |
| `er_c2` (= c₂) | `0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0` (9) |

= **225 configs/alg × 3 algs = 675 RunIDs** per `*_er_tune` expr.

(Note: a previous version of these exprs also looped over `power_mean_p` for
the MENTS-family. `MentsManager::power_mean_p` is **not read** by any code —
the loop produced 4 redundant runs per config and was dropped.)

**Sailing only** also injects `default_q_value = -200` into MENTS-family
params — sailing has only negative rewards, so default-Q=0 makes unvisited
actions wildly optimistic and degenerates MENTS into BFS. The grid is
otherwise the same.

---

## 5. Compute cost & how to shrink the grid

Per `*_er_tune` expr (UCT branch + MENTS branch combined), and the paired
`*_baselines` expr that ER is compared against:

| Stage | Per-env RunIDs | Per-run trials | Per-run repeats | Threads |
|---|---:|---:|---:|---:|
| `*_er_tune` (063 / 093 / 103)        | 432 + 675 = **1107** | 150 000 | 10 | 16 |
| `*_baselines` (064 / 094 / 104)      | 9                    | 150 000 | 10 | 16 |

ER tune and the paired baselines run at the **same horizon** (150k × 10), so
the comparison plot has matched x-extents and the ER hyperparameter selection
horizon equals the displayed horizon (no "tuned-for-different-horizon" bias).

Each `RunID` = 10 replicates × 150 000 trials with a thread pool of 16. Wall
clock varies by env (sailing trials are short, taxi ones much longer because
`max_trial_length = 200`). On a 16-core machine expect roughly **a couple of
days** per `*_er_tune` expr; the paired `*_baselines` expr finishes in
minutes.

**Ways to shrink before launching:**

1. **Coarse → fine sweep.** Drop the outer `er_c2` to `{0.1, 1.0, 10.0}`
   first (3 values instead of 9 — cuts to 1/3) and re-do a finer sweep
   around the winner.
2. **Reduce repeats during tuning.** 10 reps gives reasonably tight rankings;
   you can drop to 5 if `find_best_hyperparams.py` shows clear separation
   (the `spread` column tells you whether the winner is robust). Bump back
   to 10 (or higher) for the final paired baselines / test run that you
   actually plot.
3. **Restrict to the algorithms you actually care about.** If you only need
   ER-MENTS and ER-Fixed-Depth-UCT, comment out the others in
   `run_id.cpp` — saves the corresponding fraction of RunIDs.
4. **Halve `num_trials`.** `find_best_hyperparams.py` uses the value at the
   **final** num_trials only, so 75 000 vs 150 000 is twice as fast and
   usually picks the same winner if convergence is monotone (verify by
   plotting one or two CSVs).
5. **Batch and resume.** The eval CSV is opened with `ios::out` (overwrite)
   per RunID, so you can split the grid across multiple invocations by
   editing `run_id.cpp` to push only a subset of `RunID`s, then merge later.

If you want a "smoke test" first, the existing `060_fl12_er_uct_smoke` and
`061_fl12_er_ments_smoke` exprs run ~10 configs each in a few minutes —
useful to confirm the pipeline before spending real compute.

---

## 6. Picking & using the winners

`find_best_hyperparams.py <results_dir>` output looks like:

```
Algorithm: ER_MENTS
  Best Value (mean @ max trials): 0.4321
  Best File: results/.../063_fl12_er_tune/er-ments/eval_temp=0.001,epsilon=1,er_c2=0.3.csv
  Optimal Hyperparams: temp=0.001, epsilon=1, er_c2=0.3
```

For each ER algorithm, write down `(temp, epsilon, er_c2)` (or
`(bias, er_c2, power_mean_p)` for UCT-family) and paste them into the
follow-up `065_fl12_er_test` expr (Step 5).

**Sanity-check the winners.** Before you commit to those hyperparameters:

- Open the winner's `eval_*.csv` and check the convergence curve isn't
  noisy or still climbing at the final trial — if it is, you've under-trained.
- Cross-check across multiple replicates: the script averages 5 `mc_eval_mean`
  values; if the per-replicate spread is huge (e.g. one replicate at 0.9 and
  four at 0.1), the winner may just be a lucky seed. Plot the curves and look
  at `mc_eval_std` instead of `mc_eval_mean`.
- Look at the second- and third-best configs. If `er_c2 = 0.3` wins by 0.001
  over `er_c2 = 1.0`, the ranking is essentially noise; pick the simpler /
  more interpretable value.

---

## 7. Sailing-specific note

Sailing's reward is `-1 - tack` (`sailing_env.cpp:149`), so all returns are
negative. MENTS-family algorithms must be initialised with a pessimistic
`default_q_value` (we use **−200**, matching `S6_091_HPS` / `S6_092_TEST`)
so that unvisited actions don't look better than visited ones with
necessarily-negative Q. Both `S6_093_ER_TUNE` and `S6_094_BASELINES` set
this — if you fork or copy them, keep that line.

**Score interpretation:** "best `mc_eval_mean`" = least-negative.
`find_best_hyperparams.py` already maximises, which is correct for sailing
(closer to 0 is better).

---

## 8. Known limitations of the current pipeline

- **`alias_use_caching` and `use_max_heap` are off everywhere.** ER-RENTS
  explicitly throws if either is enabled (the unaugmented-prior fix is
  implemented only for the standard non-alias non-max-heap path). If you
  flip those flags later, you will need to extend `ERRentsDNode::select_action`
  to also fix the alias and max-heap paths.
- **`use_avg_return` is off everywhere.** If you flip it on for ER-MENTS or
  ER-TENTS, the soft-value backup will leak ER through `MentsDNode::backup_entropy`
  (a known latent bug, deliberately left because it's not exercised).
- **`MentsManager::power_mean_p` is dead.** The MENTS family doesn't use
  power-mean backups in this codebase. The `power_mean_p` knob only affects
  UCT/Fixed-Depth-UCT (via `UctDNode::backup_average_return`). Don't bother
  setting it on MENTS-family params.
- **Plot script is a tag switchboard.** If you add a new expr_id (Step 5),
  you'll need to add a matching block in `plot.py` or extend the `all_figs`
  branch.
- **Re-running an expr overwrites previous results** (eval CSV opened with
  `ios::out`). Back up `results/<env>/<instance>/<expr>/` before re-running
  if you want to keep both.
- **`num_threads > 1` makes runs non-deterministic.** This is by design;
  averaging over `num_repeats` is the variance-reduction mechanism.

---

## 9. Automation — `run_experiment.py`

For end-to-end execution there's a Python orchestrator that wires Steps 1–6 together:

```bash
# full pipeline for one env (~hours-to-days for the tune step)
python run_experiment.py fl12

# all three envs sequentially
python run_experiment.py fl12 s6 tx5

# already ran tune+baselines; just refresh the plot
python run_experiment.py fl12 --plot-only

# already built thts-run-toy-env
python run_experiment.py s6 --skip-build
```

What it does per env:

1. `make -j4 thts-run-toy-env` (unless `--skip-build`)
2. `./thts-run-toy-env <tune_expr>` (unless `--skip-tune` or `--plot-only`)
3. `./thts-run-toy-env <baseline_expr>` (unless `--skip-baselines` or `--plot-only`)
4. Imports `find_best_hyperparams.find_best_hyperparams` in-process to pick
   the top-1 winner per ER algorithm.
5. Writes `<results_root>/<tune_expr>/best_er_params.json` with the full
   top-K results per algorithm (file path, basename, mean, spread, num_reps,
   params).
6. Invokes `python plot.py compare_<env>_tune --er-best-json=<path>` so the
   plot uses the freshly-picked winners (no need to hand-edit `plot.py`).

The flag `--er-best-json=PATH` works on `plot.py` directly too:

```bash
python find_best_hyperparams.py results/frozen_lake_env/FL_8x12/063_fl12_er_tune \
    --json-out fl12_winners.json
python plot.py compare_fl12_tune --er-best-json=fl12_winners.json
```

This is what the orchestrator does internally; useful if you want to inspect
the JSON before plotting, or share winners across machines.

The hardcoded `FL12_ER_BEST` / `S6_ER_BEST` / `TX5_ER_BEST` dicts inside
`plot.py` are still consulted as a fallback when no JSON is supplied — useful
if you want to cherry-pick #2 over #1 (e.g. when winners are within noise of
each other).

> The orchestrator does not run anything in the background. Tune steps may
> take hours-to-days; launch under `nohup` / `tmux` / `screen` so the run
> survives a disconnected SSH session.

---

## 10. Quick reference

Automated path (recommended):

```bash
python run_experiment.py fl12 s6 tx5      # full pipeline, all three envs
python run_experiment.py fl12 --plot-only # refresh plot from existing results
```

Manual path (if you need to drive a single step):

```bash
make thts-run-toy-env
./thts-run-toy-env 063_fl12_er_tune                                # 1107 RunIDs, slow
./thts-run-toy-env 064_fl12_baselines                              # 9 RunIDs, fast
python find_best_hyperparams.py \
    results/frozen_lake_env/FL_8x12/063_fl12_er_tune \
    --json-out results/frozen_lake_env/FL_8x12/063_fl12_er_tune/best_er_params.json
python plot.py compare_fl12_tune \
    --er-best-json=results/frozen_lake_env/FL_8x12/063_fl12_er_tune/best_er_params.json
```

Optional follow-up — evaluate the tuned ER on the held-out test instance
(see Step 5 above): add `065_fl12_er_test` (or `095_s6_er_test`,
`105_tx5_er_test`) to `run_id.cpp`, then run it the same way.

---

## 11. Where to look in the code

- `src/toy_envs/run_id.cpp` — every expr_id branch, hyperparameter grids.
- `include/toy_envs/run_id.h` — string IDs for envs, instances, algorithms,
  parameters, and exprs.
- `src/toy_envs/run_toy.cpp` — execution loop, eval CSV format, file
  layout under `results/`.
- `find_best_hyperparams.py` — winner-selection logic (mean of `mc_eval_mean`
  at final `num_trials`).
- `plot.py` — tag-driven plotting blocks.

For algorithm internals see `include/algorithms/<family>/README.md` and the
top-level `CLAUDE.md`.
