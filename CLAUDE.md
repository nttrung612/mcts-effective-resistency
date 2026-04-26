# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

C++17 implementation of **Trial-based Heuristic Tree Search (THTS)** with a family of MCTS algorithms (UCT, PUCT, MENTS/RENTS/TENTS/DENTS/EST, plus "ER" exploration-rate variants and DB-MENTS) evaluated on toy MDPs (D-chain, Frozen Lake, Sailing, Taxi). Python scripts (`plot.py`, `find_best_hyperparams.py`) consume the CSVs produced by the C++ runner.

The repo also has a Go (game) experiment path against KataGo, but **the default `Makefile` does not build it** — see `MakefileSupp` and the README's "Go Stuff" section if you need it. Day-to-day work happens in the toy-env path.

## Build

First-time setup (gtest is a git submodule and must be built before `make` works):

```
git submodule init && git submodule update
cd external/googletest && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.
make && make install
```

If your default `g++` is not the right compiler (e.g. clang on macOS), pass `-DCMAKE_C_COMPILER=... -DCMAKE_CXX_COMPILER=...` to cmake **and** edit `CXX` at the top of `Makefile` to match — both must agree or linking fails.

Then from the repo root:

- `make` — builds both `thts-test` and `thts-run-toy-env` (default target).
- `make thts-run-toy-env` — just the experiment runner.
- `make thts-run-toy-env-debug` — same with `-g`.
- `make thts-test` / `make thts-test-debug` — gtest binary.
- `make clean` — remove `bin/` and target binaries.

There are three Makefiles: `Makefile` (local dev, the one to edit), `MakefileSupp` (used when packaging supplementary material — strips `-O2`, otherwise similar), and historically `MakefileJade` for a remote server. Keep `Makefile` and `MakefileSupp` in sync when adding new source dirs to `SOURCES`.

## Run experiments

The runner is dispatched purely by experiment id:

```
./thts-run-toy-env <expr_id>
```

Where `<expr_id>` matches one of the `*_EXPR_ID` constants in `include/toy_envs/run_id.h`. Common ones:

- Frozen Lake: `051_fl12_hps`, `052_fl12_test`, `063_fl12_er_tune`, `064_fl12_baselines`
- Sailing: `091_s6_hps`, `092_s6_test`, `093_s6_er_tune`, `094_s6_baselines`
- Taxi: `101_tx5_hps`, `102_tx5_test`, `103_tx5_er_tune`, `104_tx5_baselines`
- D-chain: `001_len_10`, `003_len_20`, `021_len_10_main_paper`, ...

Output lands at `results/<env_id>/<env_instance_id>/<expr_id>/<alg_id>/{eval_*.csv, log_*_NNN.csv, tree_*.txt}` (paths constructed in `src/toy_envs/run_toy.cpp`). The runner expands one `expr_id` into many `RunID` configs (alg × hyperparam grid × replicates) and runs them sequentially; replicates inside one `RunID` use `num_threads` workers via `ThtsPool`.

### Tests

```
./thts-test                                        # run all
./thts-test --gtest_filter='UctTest.*'             # by suite
./thts-test --gtest_filter='*-*todo*'              # skip placeholder tests
```

Many tests intentionally have `todo` in their names and are expected to fail — only treat non-`todo` failures as real.

### Analysis

- `python plot.py <expr_id>` — produces plots into a `plots/` folder (matplotlib + seaborn). `plot.py` is a giant `if "<tag>" in sys.argv` switchboard; pass either an explicit tag (e.g. `000_fig_fl`) or an `expr_id` that matches one of the hardcoded blocks. `all` triggers a curated paper-figure set, `all_figs` is broader.
- `python find_best_hyperparams.py <results_dir> [<results_dir>...]` — point at e.g. `results/frozen_lake_env/FL_8x12_test/052_fl12_test`; reads each algorithm's `eval_*.csv`, picks the row with the largest `num_trials`, averages `mc_eval_mean` across replicates, and prints the winning hyperparam set per algorithm.

## Architecture

### THTS trial loop

A `ThtsPool` (in `include/thts.h`) is a thread pool whose workers run trials until a wall-time or trial-count budget is hit. Each trial:

1. Sample a `ThtsEnvContext` (`include/thts_env_context.h`) — a string→`shared_ptr<void>` bag passed to every node call in this trial. Used for things like belief-state sampling, multi-objective scalarisations, or **passing parent state down to avoid lock cycles** (RENTS uses this — accessing the parent's distribution directly while holding a child lock would deadlock).
2. Walk down from the root: `decision_node.visit() → select_action() → chance_node.visit() → sample_observation() → next decision_node`, until a new node is created or a sink/max-depth is hit.
3. Walk back up calling `backup()` on each visited node, bottom-first.

`ThtsManager` (`include/thts_manager.h`) is the per-search "globals" object — env pointer, max depth, RNG seeds, and algorithm-specific options. Subclasses like `UctManager`, `MentsManager`, `DentsManager` add hyperparameters (bias, temperature, epsilon, default Q-value, power-mean p, ...).

The pool locks each `ThtsDNode` / `ThtsCNode` around its function calls, so node implementations do **not** take their own locks but **must** avoid taking other nodes' locks while holding their own. When upward data flow is needed, push it through `ThtsEnvContext` instead.

### `_itfc` convention and templates

The base `ThtsDNode` / `ThtsCNode` work in a type-erased "interface" world (`Action` / `Observation` are passed as `shared_ptr<const IntPairAction>` etc.) so that `ThtsPool` doesn't need to know the concrete types. Concrete algorithms get the typed view via the templates in `include/templates/`:

- `create_child_node` (typed) → `create_child_node_itfc` (final, handles transposition tables) → `create_child_node_helper_itfc` (virtual, dispatch to subclass) → `create_child_node_helper` (typed, constructs the node).
- When building a new algorithm, copy a template, find-and-replace the placeholder typenames, and only override the typed `_helper` and the algorithm-specific `visit/select/sample/backup` entries.

`include/algorithms/` is split by family: `uct/` (UCT, PUCT, AlphaGo PUCT, ER variants, fixed-depth UCT), `ments/` (MENTS, ER-MENTS, DB-MENTS, plus subdirs for `dents/`, `rents/`, `tents/`), `est/` (EST), `common/` (shared dynamic-programming and entropy backup nodes, `power_mean`, `decaying_temp`, `max_heap`, `emp_node`).

### Adding a new experiment

1. Add a `XYZ_EXPR_ID` constant in `include/toy_envs/run_id.h`.
2. Add an `if (expr_id == XYZ_EXPR_ID) { ... }` block in `get_run_ids_from_expr_id` in `src/toy_envs/run_id.cpp` that builds `RunID` entries (env, alg list, hyperparam grid, trials, replicates, threads) and pushes them onto the result vector.
3. New algorithm hyperparams need to be wired through `RunID::get_thts_manager` / `get_root_search_node` / `get_logger` in the same file — those switch on `alg_id` and read `run_id.alg_params[PARAMS_ID_*]`.
4. If you add an algorithm in a new subdirectory, also add the `wildcard` line for its `*.cpp` to `SOURCES` in **both** `Makefile` and `MakefileSupp`.

### Toy envs

`include/toy_envs/` contains `d_chain_env.h`, `frozen_lake_env.h`, `sailing_env.h`, `taxi_env.h`, plus the `run_id.h` / `run_toy.h` glue. They subclass `ThtsEnv` and implement the typed transition / reward / observation interface that `_itfc` boilerplate forwards into.

## Notable conventions

- "**sink**" = terminal/absorbing state; "**leaf**" = any node forced to have zero children (sink, max-depth, etc.). Don't conflate them.
- Every directory under `include/` has its own `README.md` with a more detailed orientation — read those before changing code in an unfamiliar subtree.
- The branch name in the README (`xpr_go`) refers to a frozen pre-refactor snapshot used for the NeurIPS 2023 Boltzmann-exploration MCTS paper. Active development branches (e.g. `claude-code`) sit on top of that snapshot — the public THTS++ `main` branch elsewhere has had a major refactor that is **not** present here.
