---
name: SCION Simulator Guidelines
description: "Use when: understanding the SCION DQN simulator codebase, evaluation pipelines, running tests, or developing new topological features."
---

# SCION DQN Simulator Rules and Guidelines

Welcome to the `scion-dqn-sim` repository! We simulate the SCION (Scalability, Control, and Isolation on Next-generation Networks) routing architecture and evaluate RL/DQN agents against baseline path selection heuristics.

## Core Concepts & Architecture
- **SCION Properties**: Relies on Isolation Domains (ISDs), Core/Non-Core ASes, and hierarchical top-down, core-mesh beacon propagation. Peering links and Up/Core/Down Segment assembly form our path generation logic.
- **Evaluation Pipeline**: The root execution logic lives in `evaluation/`. The pipeline cascades through numbered scripts (`01_generate_topology.py` -> `06_generate_figures.py`).
- **Data Sharing**: Intermediate data is passed between phases using `.pkl` or `.json` formats within timestamped run directories (`run_YYYYMMDD_HHMMSS/`). Files include `path_store.pkl`, `link_states.pkl`, `traffic_flows.pkl`, `scion_topology.json`. 

## Best Practices
- **Run the pipeline**: You can execute the entire pipeline with `uv run python run_full_evaluation.py` from within the `evaluation/` folder.
- **Testing**: We use `pytest`. Execute `uv run python -m pytest` from the root directory. Keep the tests in `tests/` up to date with new agent logic or path simulators. 
- **Tooling**: We use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. Always install/sync changes with `uv sync` instead of `pip install`.
- **Topologies**: Use `src/topology/` to interact with or adapt JSON/NetworkX graphs. We support BRITE configurations via `setup_brite.sh`.

## AI Agents
When extending code, consider the following structural divisions:
- `src/rl/`: Contains the DQN components mapping SCION 5-dimensional State spaces to Actions (`dqn_agent_simple.py` and `dqn_agent_enhanced.py`).
- `src/beacon/` & `src/simulation/`: Houses the SCION Path Builders, ensuring structural constraints (Up -> Core -> Down + Peering) when routing between source and destination endpoints.
- `evaluation/`: Coordinates episodic simulations over traffic snapshots and extracts resulting tables/charts.
- **Code modification constraints**: Favor composing functions and respecting backwards compatibility if adding parallel methods. Do not blindly overwrite baseline methods unless instructed.

## Helpful Pointers
- Start with `README.md` to see configuration knobs (like `EVAL_BRITE_N_NODES=45`).
- Look at `evaluation/_common.py` for shared logic between the sequential evaluation steps.