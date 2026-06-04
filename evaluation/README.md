# DQN Path Selection Evaluation

Numbered pipeline scripts (`01`–`06`) orchestrated by `run_full_evaluation.py`.

## Layout

| Location | Role |
|----------|------|
| `evaluation/*.py` | Thin pipeline entry points (CLI + orchestration) |
| `evaluation/topology_defaults.yaml` | Step 01 topology configuration |
| `src/pipeline/` | Run-directory resolution, subprocess runner, figure styling, `dqn_train_cli` |
| `src/simulation/` | Path store, evaluation env, traffic simulation, run context |
| `src/rl/path_selection_train.py` | `train_flat_dqn`, `train_scoring_dqn` |
| `src/topology/eval_config.py` | YAML config loading and BRITE generation |

## Run pipeline

```bash
cd evaluation
uv run python run_full_evaluation.py

# Reuse topology + traffic from an existing run:
uv run python run_full_evaluation.py --from-step 4 --run-dir run_YYYYMMDD_HHMMSS
```

## Topology visualization (optional)

```bash
uv run python -m src.visualization.topology_cli
uv run python -m src.visualization.topology_cli run_YYYYMMDD_HHMMSS --mode simple
```

## Step 03 traffic state

Step 03 writes a **compact** `link_states.pkl` (`link_hourly_v1`: hourly link-level metrics only). Path metrics for each `(src, dst)` are computed on demand during training/eval, so large topologies (e.g. 100 ASes / ~10k pairs) stay within memory. Re-run step 03 if you have an old multi-gigabyte `link_states.pkl` from a previous version.

## Step 04 training

- **`04_train_all_models.py`** — pipeline default; loads `link_states.pkl` once, trains all checkpoints.
- **`04_train_*.py`** — thin wrappers around `src.pipeline.dqn_train_cli` (shared `--config-json`, `--checkpoint`, `--stats-json` where applicable).

## Observations

- **Flat DQN** (`04_train_dqn.py`, `04_train_simple_dqn.py`): `env.observe_flat()` → 5-D vector
- **Path-scoring DQN** (`04_train_scoring_*.py`): `env.observe_scoring()` → global + per-path features

Probe penalty is normalized per probe so baselines that probe all paths are not over-penalized vs selective RL.

Reward and goodput normalization are computed inside `env.step()` / `env.apply_action()` so training and evaluation stay aligned.
