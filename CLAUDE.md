# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A SCION AS-level network simulator used to evaluate Deep Q-Network (DQN) path-selection
agents against heuristic baselines. The work is a thesis project; some components are
work-in-progress.

## Setup

Requires the BRITE topology generator (a git submodule) and [`uv`](https://docs.astral.sh/uv/)
for the Python environment.

```bash
git submodule update --init --recursive   # pulls external/brite
./setup_brite.sh                           # builds the BRITE Java generator
uv sync --extra dev                        # creates .venv, installs package editable + pytest
```

Run everything through the project venv: `uv run python ...` (or activate `.venv`).
Use `uv sync` to change dependencies, never `pip install`. `requirements.txt` is a mirror
of `pyproject.toml` kept for reference only.

## Commands

```bash
# Full evaluation pipeline (creates evaluation/run_YYYYMMDD_HHMMSS/)
cd evaluation && uv run python run_full_evaluation.py
uv run python run_full_evaluation.py --run-dir run_20260101_120000   # reuse a run dir
uv run python run_full_evaluation.py --from-step 4                   # resume partway

# Tests (from repo root)
uv run python -m pytest                    # all
uv run python -m pytest tests/test_evaluation_env.py::test_name   # single test

# Lint / format (dev extras)
uv run black src tests evaluation
uv run flake8 src tests evaluation
```

Fast smoke runs: `EVAL_BRITE_N_NODES=45 uv run python run_full_evaluation.py` (small topology),
and `DQN_TRAIN_EPISODES=N` to cap training length. On Windows PowerShell set env vars with
`$env:EVAL_BRITE_N_NODES=45` before the command rather than the inline `VAR=val` prefix.

## Architecture

The key split: **`evaluation/` holds thin numbered CLI scripts; all real logic lives in `src/`.**
Each script `0N_*.py` takes a run directory as `argv[1]` (or defaults to the latest `run_*`
in the cwd) and imports from `src`. The orchestrator `run_full_evaluation.py` runs the steps
as subprocesses via `src/pipeline/run_dirs.py`.

Steps pass data through files inside a timestamped **run directory** (`run_YYYYMMDD_HHMMSS/`):

1. **01_generate_topology** → `topology/scion_topology.{json,pkl}` from BRITE, with SCION roles
   (ISDs, core/non-core ASes) assigned. Reads `evaluation/topology_defaults.yaml`. Modules: `src/topology/`.
2. **02_run_beaconing** → `path_store.json`, `selected_pair.json`. Simulates SCION control plane
   (core-mesh + top-down intra-ISD PCB propagation; peer links excluded from beaconing) and
   assembles Up→Core→Down(+peering) paths. Modules: `src/beacon/`, `src/simulation/beacon_pipeline.py`, `path_builder.py`.
3. **03_simulate_traffic** → `link_states.pkl` (compact `link_hourly_v1` format), `traffic_flows.pkl`.
   28 days of hourly demand for every routable pair plus background traffic, aggregated **per link**
   so paths share bottlenecks. Modules: `src/simulation/link_traffic_sim.py`, `traffic_config.py`.
4. **04_train_\*** → `*.pth` checkpoints. Multi-pair DQN training on the first 14 days. Stateful
   episodes (advances `hour_idx` per step), action masking for variable per-pair path counts.
   Modules: `src/rl/`. `04_train_all_models.py` runs every variant; individual `04_train_*.py` train one.
5. **05_evaluate_methods** → `evaluation_results.json`. Runs DQN + baselines over the last 14 days.
   Modules: `src/simulation/evaluation_env.py`, `src/baselines/`.
6. **06_generate_figures** → `figures/*.png`. Module: `src/pipeline/figures.py`.

**Integration hub:** `src/simulation/` — `run_context.py` provides `make_env()`,
`load_run_context()`, and artifact validation; `evaluation_env.py` defines
`EvaluationPathSelectionEnv` shared by training (04) and evaluation (05).

### DQN model families (`src/rl/`)

There are several agent architectures, each with its own `04_train_*.py` script and `.pth` file:
- **flat** (`dqn_agent_simple.py`): 5-D global state, discrete masked path index.
- **scoring** (`dqn_agent_scoring_*.py`): per-path Q-values over `{global, paths}` observations,
  handling a variable number of paths. The enhanced/dueling scoring agent is the recommended one.
- **conditional** (`dqn_agent_scoring_conditional.py`): scoring agent whose global state also encodes
  the reward weights, for multi-objective training (`reward_profiles.py`).

Training entry points are `train_flat_dqn()` / `train_scoring_dqn()` / `train_conditional_scoring_dqn()`
in `path_selection_train.py`.

### Baselines (`src/baselines/`)

Each selector implements `select_path(paths, metrics, flow, state) -> int`: shortest-path, widest-path,
lowest-latency, ECMP, SCION-default, random. Register new ones in `05_evaluate_methods.py`.

## Key concepts

- **Selective probing accounting**: the DQN probes only its chosen path; probe overhead (latency probes
  10ms + 0.5ms/hop, bandwidth probes 100ms + 20ms/hop) is reported **separately** from measured path
  latency. When comparing methods, latency tables exclude probe cost — don't conflate them.
- **Reward**: composite `r = 2·(w1·G + w2·T) − 1` where `G` is goodput normalized against an
  auto-detected per-topology cap and `T` is a link-trust score. See `EvaluationPathSelectionEnv.compute_reward()`.
- **Run-dir resolution**: most scripts pick the lexicographically latest `run_*` when no path is given.
  When invoking a single step manually, pass the run dir explicitly to avoid acting on the wrong run.

## Extending

- **New traffic behaviour**: edit `TrafficSimConfig` / `link_traffic_sim.py`, then re-run steps **03 and 04**
  (training depends on the regenerated `link_states.pkl`).
- **New baseline / new DQN**: see the "Extending the codebase" section of `src/README.md` for the exact
  registration points. Favor adding parallel methods over overwriting existing baselines.

## Configuration knobs

Behaviour is tuned largely through environment variables, documented in `src/README.md`
("Environment variables") and `README.md` ("Key knobs"). Common ones: `EVAL_BRITE_N_NODES`,
`DQN_TRAIN_EPISODES`, `DQN_TRAIN_PAIR_CAP`, `TRAFFIC_N_JOBS`, `BEACON_MAX_SEGMENTS_PER_ORIGIN`,
and the `TRAFFIC_*` / `DQN_*` families.

## Further reading

- `README.md` — install + full pipeline quick start, output-file table, programmatic examples.
- `src/README.md` — module-by-module map and observation/reward dimensions.
- `AGENTS.md` — contributor guidelines.
