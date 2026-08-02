# DQN Path Selection Evaluation

Numbered pipeline scripts (`01`–`06`) plus optional multi-profile eval, orchestrated by `run_full_evaluation.py`.

## Layout

| Location | Role |
|----------|------|
| `evaluation/*.py` | Thin pipeline entry points (CLI + orchestration) |
| `evaluation/topology_defaults.yaml` | Step 01 topology configuration |
| `src/pipeline/` | Run-directory resolution, subprocess runner, figure styling, `dqn_train_cli` |
| `src/simulation/` | Path store, evaluation env, traffic simulation, run context |
| `src/rl/path_selection_train.py` | `train_flat_dqn`, `train_scoring_dqn`, `train_conditional_scoring_dqn` |
| `src/topology/eval_config.py` | YAML config loading and BRITE generation |

## Run pipeline

```bash
cd evaluation
uv run python run_full_evaluation.py

# Reuse topology + traffic from an existing run:
uv run python run_full_evaluation.py --from-step 4 --run-dir run_YYYYMMDD_HHMMSS

# Cap training episodes for all step-04 trainers:
uv run python run_full_evaluation.py --train-episodes 200
```

Default step order (`src/pipeline/run_dirs.py`):

| Step | Script | Output (under `run_*/`) |
|------|--------|---------------------------|
| 01 | `01_generate_topology.py` | `topology/scion_topology.json` |
| 02 | `02_run_beaconing.py` | `path_store.pkl` |
| 03 | `03_simulate_traffic.py` | `link_states.pkl`, `traffic_flows.pkl` |
| 04 | `04_train_all_models.py` | DQN checkpoints (see below) |
| 05 | `05_evaluate_methods.py` | `evaluation_results.json` |
| — | `eval_multi_reward_comparison.py` | `multi_reward_comparison.json` |
| 06 | `06_generate_figures.py` | `figure1_*.png`, `figure2_*.png`, `figure3_*.png`, `figure4_*.png` |

Pass the run directory as the first positional argument to any step script, or omit it to use the latest `run_*` in `evaluation/`.

## Topology visualization (optional)

```bash
uv run python -m src.visualization.topology_cli
uv run python -m src.visualization.topology_cli run_YYYYMMDD_HHMMSS --mode simple
```

## Step 03 — traffic state

Step 03 writes a **compact** `link_states.pkl` (`link_hourly_v1`: hourly link-level metrics only). Path metrics for each `(src, dst)` are computed on demand during training/eval, so large topologies (e.g. 100 ASes / ~10k pairs) stay within memory.

Re-run step 03 if you have an old multi-gigabyte `link_states.pkl` from a previous version (legacy format stored per-path arrays for every pair).

Optional inspection:

```bash
uv run python inspect_traffic.py run_YYYYMMDD_HHMMSS
```

## Step 04 — training

**Pipeline default:** `04_train_all_models.py` loads `link_states.pkl` once and trains every variant in one process.

**Individual trainers** (thin wrappers around `src.pipeline.dqn_train_cli`):

| Script | Checkpoint |
|--------|------------|
| `04_train_dqn.py` | `dqn_model.pth` |
| `04_train_simple_dqn.py` | `dqn_simple_model.pth` |
| `04_train_scoring_dqn.py` | `dqn_scoring_simple_model.pth` |
| `04_train_scoring_enhanced_dqn.py` | `dqn_scoring_enhanced_model.pth` |
| `04_train_conditional_dqn.py` | `dqn_conditional_scoring_model.pth` |

Shared CLI flags: `--config-json`, `--checkpoint`, `--stats-json` (where applicable).

```bash
uv run python 04_train_conditional_dqn.py run_YYYYMMDD_HHMMSS
uv run python 04_train_scoring_enhanced_dqn.py run_YYYYMMDD_HHMMSS --config-json my_hp.json
```

### Training environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `DQN_TRAIN_EPISODES` | (scaled) | Fixed episode count; overrides pair-pool scaling |
| `DQN_TRAIN_PAIR_CAP` | `64` | Caps effective pair pool when scaling episode count |
| `DQN_GRADIENT_EVERY` | `4` | Run `replay()` every N environment steps |
| `DQN_CONDITIONAL_PROFILES` | `distinctive` | Conditional training profiles: `distinctive`, `legacy` (eval presets), or `all` |
| `DQN_CONDITIONAL_EPISODE_MULT` | `1.25` | Extra episodes for conditional vs other scorers |
| `TRAFFIC_N_JOBS` | (see step 03) | Parallelism for traffic simulation |

Episode count otherwise scales as `max(50, min(20000, 200 × pair_cap) / 24)` unless `DQN_TRAIN_EPISODES` is set.

## Observations and rewards

All methods share `EvaluationPathSelectionEnv` (`src/simulation/evaluation_env.py`). Reward is computed in `env.apply_action()` / `env.step()` so training and evaluation stay aligned.

| Agent | Observation | Global dim |
|-------|-------------|------------|
| Flat DQN | `observe_flat()` | 5 |
| Path-scoring DQN | `observe_scoring()` → `global` + `paths` | 7 + N×7 |
| Conditional DQN | `observe_scoring_conditional()` | 12 (7 + 5 weight encoding) |

Per-path features (7-D): latency, loss, hops, relative bandwidth, utilization, static bandwidth, trust.

Probe penalty is normalized per probe so baselines that probe all paths are not over-penalized vs selective RL.

### Conditional DQN (multi-objective)

- **Inference:** set `env.reward_weights` (or use named profiles from `src/rl/reward_profiles.REWARD_PROFILES`), then `observe_scoring_conditional()`.
- **Training:** samples from `DISTINCTIVE_REWARD_PROFILES` by default (extreme bandwidth / loss / delay / probe objectives) with stratified scheduling; eval still uses the six standard `REWARD_PROFILES`.
- **Architecture:** weight-FiLM dueling head (`ConditionalPathScoringDQNAgent`) so reward weights modulate per-path Q-values, not only a shared value offset. Checkpoints record `architecture` and `weight_encoding`; legacy concat checkpoints load via `load_conditional_scoring_agent()`.

Scoring enhanced is trained on a single **balanced** weight vector; conditional is trained across multiple objectives.

## Step 06 — figures

`06_generate_figures.py` reads `evaluation_results.json` and, when present, `multi_reward_comparison.json`.

| Output | Description |
|--------|-------------|
| `figure1_probe_overhead.png` | Probe overhead and selection time |
| `figure2_path_reward.png` | Path reward by method (**balanced** profile) |
| `figure2_path_reward_<profile>.png` | Same plot for each eval profile (`throughput`, `trust_quality`, …) when multi-reward JSON exists |
| `figure3_probe_breakdown.png` | Latency vs bandwidth probes per selection |
| `figure4_multi_reward_heatmap.png` | Method × profile reward heatmap |

Per-profile Figure 2 requires `eval_multi_reward_comparison.py` (included in `run_full_evaluation.py` before step 06). Without it, only `figure2_path_reward.png` is produced from step 05 stats.

## Optional analysis scripts

Not part of the numbered pipeline unless invoked via `run_full_evaluation.py` (multi-reward step is included there).

```bash
# Per-method mean reward under each eval profile (DQN variants + all baselines)
uv run python eval_multi_reward_comparison.py run_YYYYMMDD_HHMMSS

# Conditional-only profile sweep
uv run python eval_conditional_dqn_rewards.py run_YYYYMMDD_HHMMSS

# Action agreement and cross-profile diversity diagnostics
uv run python analyze_conditional_sensitivity.py run_YYYYMMDD_HHMMSS
```

## Seed sweeps

Every result is reported as a mean with a 95% confidence interval over
five *training* seeds. Only torch/numpy/random are reseeded — the environment's
pair, hour and profile streams stay fixed, so each seed sees the identical stream
of training contexts and is graded on the same 10752 held-out decision contexts
(32 pairs × 336 hours). Four rungs are seeded: Flat DQN, the unconditioned
Scoring DQN, Value-Concat, and Two-Stream-Concat. FiLM was dropped from the
thesis and is not trained or reported.

```bash
# 1. Retrain the ladder under seeds 1..5 and re-run the ablation on each.
./run_seed_sweep.sh run_YYYYMMDD_HHMMSS

# 2. Re-run the other five results per seed (intent alignment,
#    zero-shot interpolation, path-count scaling + order invariance, probing
#    overhead, congestion ceiling). Loads the run context once for all seeds.
uv run python run_seed_result_sweep.py run_YYYYMMDD_HHMMSS

# 3. Aggregate to means with CIs and test whether each claim survives.
uv run python analyze_seed_results.py run_YYYYMMDD_HHMMSS

# 4. Redraw the six thesis figures with the spread shown, and install them.
uv run python plot_seed_figures.py run_YYYYMMDD_HHMMSS --copy-to ~/thesis-report/figures
```

| Script | Output |
|--------|--------|
| `run_seed_sweep.sh` | `seeds/seed<N>/` — checkpoints + `ablation/` |
| `run_seed_result_sweep.py` | `seeds/seed<N>/shipped/{intent,zeroshot,pathcount,probing}/` |
| `analyze_seed_variance.py` | `gap/stats/seed_variance.json`, `significance.json` (ablation) |
| `analyze_seed_results.py` | `seeds/aggregate/*.csv`, `claims.json` (the other five results) |
| `plot_seed_figures.py` | `seeds/aggregate/figures/p1eval_*.png` |

`run_seed_result_sweep.py` is idempotent — a study whose CSVs already exist is
skipped, so it can be re-run to fill in a newly added seed. Pass `--force` to
recompute.

## Tests

From the repository root:

```bash
uv run python -m pytest
```

Conditional DQN tests: `tests/test_conditional_scoring_dqn.py`.
