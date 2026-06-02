# Project Overview: SCION DQN Simulator

A simulator for SCION (Scalable, Control-free, Isolation ON) networks with Deep Q-Network (DQN) based path selection optimization. The project integrates topology generation, a realistic SCION control plane (beaconing), traffic/link dynamics simulation, and reinforcement learning for path selection.

## Core Components

- **Topology Generation (`src/topology/`)**: Uses BRITE (Java) to generate AS-level topologies, converted to SCION-ish graphs (JSON/Pickle) with ISDs and core ASes.
- **Beacon Simulation (`src/beacon/`)**: Implements SCION beaconing semantics (intra-ISD top-down propagation, core-mesh beaconing) to discover candidate paths.
- **Traffic Engine (`src/traffic/`)**: Simulates 28 days of hourly traffic flows with per-link aggregation and shared bottlenecks.
- **Reinforcement Learning (`src/rl/`)**: Features `EnhancedDQNAgent` and Gym-style environments (`EvaluationPathSelectionEnv`) with selective probing support.
- **Baselines (`src/baselines/`)**: Implements Shortest Path, Widest Path, Lowest Latency, ECMP, Random, and SCION Default selection policies.

## Building and Running

### Prerequisites
- Java (for BRITE topology generator)
- [uv](https://docs.astral.sh/uv/) for Python dependency management

### Setup
1. **Initialize BRITE**: Run `./setup_brite.sh` to initialize submodules and build the BRITE JAR.
2. **Install Dependencies**: Run `uv sync --extra dev` from the root directory.

### Execution
The project uses a 6-step evaluation pipeline orchestrated by `evaluation/run_full_evaluation.py`.

```bash
cd evaluation
uv run python run_full_evaluation.py
```

This creates a timestamped `run_YYYYMMDD_HHMMSS/` directory containing all artifacts (topology, paths, traffic, model, results).

### Key Commands
- **Run all tests**: `uv run python -m pytest`
- **Visualize Topology**: `uv run python -m src.visualization.topology_cli <run_dir> --mode full`
- **Run individual step**: `uv run python 04_train_dqn.py <run_dir>` (re-uses existing run artifacts)

## Development Conventions

- **Evaluation Pipeline**: Experiments must follow the numbered scripts in `evaluation/`. Each script accepts a run directory as its first argument and uses `src.pipeline.run_dirs.resolve_run_dir()` for consistency.
- **Artifact Management**: All results and intermediate data (pickles, JSONs, models) are stored within the specific `run_*` directory to ensure experiment isolation.
- **SCION Semantics**: Strictly adhere to SCION path selection and beaconing rules (e.g., top-down propagation, core-mesh beaconing).
- **Probing Accounting**: Maintain the separation between measured path metrics and probe overhead (`env.last_probe_cost_ms`) to ensure fair comparisons.
- **State/Reward Consistency**: If modifying `EnhancedDQNAgent` or `EvaluationPathSelectionEnv`, ensure state featurization and reward logic remain consistent across training (`04`) and evaluation (`05`) steps.
- **Imports**: Use absolute imports starting with `src.`. Pipeline scripts in `evaluation/` import `_bootstrap` first so the repo root is on `sys.path`.

## Key Files & Directories

- `evaluation/run_full_evaluation.py`: Main entry point for the experiment pipeline.
- `src/pipeline/`: Run directory resolution, subprocess execution, figure styling.
- `src/simulation/evaluation_env.py`: The lightweight RL environment used for pipeline training/evaluation.
- `src/rl/dqn_agent_enhanced.py`: The primary DQN implementation.
- `src/beacon/beacon_sim_v2.py`: The SCION-realistic beacon simulator.
- `topology/scion_topology.json`: The standard graph format used by most components.
