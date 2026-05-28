# DQN Path Selection Evaluation

Numbered pipeline scripts (`01`–`06`) plus shared modules:

| Module | Role |
|--------|------|
| `path_selection.py` | Thin re-exports from `src.simulation` / `src.rl` |
| `train_lib.py` | Thin re-export of `src.rl.path_selection_train` |
| `src/simulation/evaluation_env.py` | Unified env: probe, observe, reward, step |
| `src/simulation/run_context.py` | Load run dirs, `make_env()` |
| `src/rl/path_selection_train.py` | `train_flat_dqn`, `train_scoring_dqn` |

## Run pipeline

```bash
cd evaluation
uv run python run_full_evaluation.py

# Reuse topology + traffic from an existing run:
uv run python run_full_evaluation.py --from-step 4 --run-dir run_YYYYMMDD_HHMMSS
```

## Observations

- **Flat DQN** (`04_train_dqn.py`, `04_train_simple_dqn.py`): `env.observe_flat()` → 5-D vector
- **Path-scoring DQN** (`04_train_scoring_*.py`): `env.observe_scoring()` → `{global: 7-D with pair embed, paths: 7-D with relative bw}`

Probe penalty is normalized per probe (`num_probes_in_step`) so baselines that probe all paths are not over-penalized vs selective RL.

Reward and goodput normalization are computed inside `env.step()` / `env.apply_action()` so training and evaluation stay aligned.
