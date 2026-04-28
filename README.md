# SCION Network Simulator for DQN-based Path Selection

A simulator for SCION networks with Deep Q-Network (DQN) based path selection optimization.

Note: This is a work in progress and some components are not yet fully implemented.

## Features

- **BRITE-based topology generation**: AS-level SCION-style graphs from the BRITE Java generator (or a pure-Python `top_down` alternative)
- **SCION-realistic control plane**: top-down (parent→child) intra-ISD beaconing, core-mesh beaconing, peer links excluded from PCB propagation, per-origin fan-out caps
- **Multi-pair traffic simulation**: 28 days of hourly foreground flows for every routable AS pair, plus randomly-drawn background traffic with **per-link aggregation** so paths share bottlenecks
- **Stateful, multi-pair RL training**: episodic environment that advances `hour_idx` per step, action masking to handle variable per-pair path counts, real per-pair link-state featurization (no hardcoded placeholders)
- **Selective probing accounting**: probe overhead is reported separately from measured latency, so result tables reflect the network rather than probe cost
- **Baseline comparisons**: Shortest path, widest path, lowest latency, ECMP, random, and SCION default selectors
- **Topology visualization**: Full dashboard or geographic map from `topology/scion_topology.json` (or topology pickle in the same folder)

## Installation

1. Clone the repository with submodules:

```bash
git clone --recursive https://github.com/netsys-lab/scion-dqn-sim.git
cd scion-dqn-sim
```

Or if you already cloned without submodules:

```bash
git submodule update --init --recursive
```

2. Set up the BRITE topology generator:

```bash
./setup_brite.sh
```

3. Install Python tooling with [uv](https://docs.astral.sh/uv/) and sync dependencies (creates `.venv` and installs the package in editable mode, including dev tools such as pytest):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or use your OS package manager
uv sync --extra dev
```

Use `uv run python ...` or `source .venv/bin/activate` so scripts use the project environment.

The legacy `requirements.txt` is kept in sync with `pyproject.toml` for reference; prefer `uv sync` for installs.

## Quick start

### Complete evaluation pipeline

The evaluation pipeline compares DQN-based path selection with the baseline methods on topologies produced in `evaluation/`.

**Run all six steps** (creates a new `evaluation/run_YYYYMMDD_HHMMSS/` directory):

```bash
cd evaluation
uv run python run_full_evaluation.py
```

**Reuse an existing run directory** (re-runs every step into the same folder; overwrite previous artifacts):

```bash
cd evaluation
uv run python run_full_evaluation.py --run-dir run_20260101_120000
```

The orchestrator and numbered scripts share helpers in **`evaluation/_common.py`** (run-directory resolution, subprocess runner, figure styling metadata for step 06).

**Steps executed:**

1. **`01_generate_topology.py`** — Reads **`evaluation/topology_defaults.yaml`** (override with **`--topology-config PATH`**). Set **`generator: brite`** (default) or **`generator: top_down`**. Writes **`topology/scion_topology.{json,pkl}`** with `node['role']` set, **`topology/topology_config_resolved.yaml`** (effective merged config), and under **`topology/`** either BRITE artifacts + three **`step*.png`** snapshots, or (top-down) a Python-built graph plus three **`step*_top_down_*.png`** snapshots when **`output.save_step_pngs`** is true.
2. **`02_run_beaconing.py`** — Runs the SCION beacon simulator (top-down intra-ISD propagation, core-mesh beaconing, PEER edges skipped). Writes **`path_store.pkl`** with paths for **multiple AS pairs** (`pair_pool`) plus a legacy **`selected_pair.json`** for the diverse "best" pair.
3. **`03_simulate_traffic.py`** — Generates 28 days of hourly foreground demand for every routable pair, draws random background traffic per hour, and aggregates load **per link** so paths share bottlenecks. Writes **`link_states.pkl`** keyed by hour and pair.
4. **`04_train_dqn.py`** — Multi-pair DQN training on the first 14 days. Stateful episodes (γ matters), action masking, real per-pair link-state featurization. Goodput cap auto-derived from path bandwidths so the reward stops saturating.
5. **`05_evaluate_methods.py`** — Baselines + DQN over the last 14 days × the pair pool. Probe cost is reported via `env.last_probe_cost_ms` so DQN and baselines are compared on the same accounting.
6. **`06_generate_figures.py`** — Comparison figures (PNG). Probe-per-selection numbers are computed from `n_selections` in the results JSON, not a hardcoded constant.

**Faster / toy BRITE size** (optional): large topologies take longer in step 1. For a smoke test, set the node count before step 1 (or before the full orchestrator, since step 1 reads this variable):

```bash
cd evaluation
EVAL_BRITE_N_NODES=45 uv run python run_full_evaluation.py
```

**Typical outputs** (under the run directory):

| File | Description |
|------|-------------|
| `topology/scion_topology.json` / `topology/scion_topology.pkl` | Topology (plus `topology/topology_config_resolved.yaml`; BRITE: `step*.png`, `brite_config.conf`, `topology.brite`; top-down: `step1_top_down_layout.png`, `step2_top_down_hierarchy.png`, `step3_top_down_peering.png` when plots enabled) |
| `path_store.pkl`, `selected_pair.json` | Paths and evaluation pair |
| `traffic_flows.pkl`, `link_states.pkl` | Traffic and link dynamics |
| `dqn_model.pth`, `training_stats.json` | Trained agent and training log |
| `evaluation_results.json` | Metrics for all methods |
| `figure1_probe_overhead.png`, `figure2_path_reward.png`, `figure3_probe_breakdown.png` | Result figures |

### Individual steps

Each numbered script accepts the run directory as **`argv[1]`**, or picks the lexicographically latest `run_*` in the current working directory (when you `cd evaluation`, that is usually the latest evaluation run).

```bash
cd evaluation

mkdir -p run_YYYYMMDD_HHMMSS   # optional if 01 creates the dir when invoked without argv

uv run python 01_generate_topology.py run_YYYYMMDD_HHMMSS
uv run python 01_generate_topology.py run_YYYYMMDD_HHMMSS --topology-config my_topology.yaml
uv run python 02_run_beaconing.py run_YYYYMMDD_HHMMSS
uv run python 03_simulate_traffic.py run_YYYYMMDD_HHMMSS
uv run python 04_train_dqn.py run_YYYYMMDD_HHMMSS
uv run python 05_evaluate_methods.py run_YYYYMMDD_HHMMSS
uv run python 06_generate_figures.py run_YYYYMMDD_HHMMSS
```

### Topology maps (optional)

After a run has `topology/scion_topology.json` (or `topology/scion_topology.pkl`), generate **topology figures** with **`evaluation/visualize_topology.py`**:

```bash
cd evaluation

# Full dashboard: main geographic map + degree / ISD / link-type panels, plus extra PNGs
uv run python visualize_topology.py run_YYYYMMDD_HHMMSS --mode full --report

# Single geographic map with AS-role and link-type legends
uv run python visualize_topology.py --mode simple

# Explicit JSON path and output directory
uv run python visualize_topology.py -t run_YYYYMMDD_HHMMSS/topology/scion_topology.json -o ./figures --mode full
```

- **`--mode full`** (default): writes **`topology_dashboard.png`**, and unless **`--no-extras`** is set, also **`isd_map.png`**, **`core_network.png`**, **`connectivity_matrix.png`** in the same folder as the dashboard.
- **`--mode simple`**: writes **`topology_geographic.png`** (one figure, legends, all AS–AS links in an undirected view).
- **`--report`**: writes **`topology_stats.txt`** (counts, link mix, connectivity).
- **`-o`**: if the path does not end with **`.png`**, it is treated as an **output directory** (appropriate files are created inside).

Implementation lives in **`src/visualization/topology_visualizer.py`** (pickle and JSON inputs, shared drawing logic).

### Tests

From the repository root:

```bash
uv run python -m pytest
```

Tests live under **`tests/`** and are configured via **`[tool.pytest.ini_options]`** in **`pyproject.toml`**. They cover topology config, traffic and link helpers, path metrics, the in-memory path store, baselines, the DQN agent smoke path, evaluation helpers, and topology plotting.

### Evaluation metrics

The evaluation step compares:

- **Reward**: Composite metric `r = 2·(w1·G + w2·T) − 1` where `G` is normalized goodput against an auto-detected per-topology cap and `T` is a link-trust score (`1 − w3·loss − w4·delay/100`)
- **Latency**: Mean and percentiles (ms) of the *measured* path latency (probe overhead is **not** added in)
- **Bandwidth**: Mbps measured on the chosen path's bottleneck
- **Probe overhead**: Count and wall-clock cost of latency vs bandwidth probes
  - Latency probes: 10 ms base + 0.5 ms per hop
  - Bandwidth probes: 100 ms base + 20 ms per hop
- **Selection time**: Wall-clock time spent inside the policy / harness for one path choice
- **Probe reduction**: DQN vs average baseline probe load (also reported as time reduction)

The DQN uses **selective probing** (one full probe of the chosen path per selection); the scripted baselines probe according to each method's needs in **`05_evaluate_methods.py`**.

### Key knobs (env vars)

- **`EVAL_BRITE_N_NODES`** — overrides BRITE node count for fast smoke runs.
- **`DQN_TRAIN_EPISODES`** — overrides the auto-scaled training episode count in `04_train_dqn.py` / `04_train_simple_dqn.py`.
- **`BEACON_INTRA_MAX_POPS`**, **`BEACON_MAX_INTRA_SEGMENTS_PER_ISD`**, **`BEACON_MAX_SEGMENTS_PER_ORIGIN`** — bound beacon-simulator work on dense topologies.

### Programmatic examples

**BRITE configuration file:**

```python
from pathlib import Path
from src.topology.brite_cfg_gen import BRITEConfigGenerator

gen = BRITEConfigGenerator()
gen.generate(Path("out.conf"), n_nodes=50, seed=42)
```

**Render a topology JSON to PNG:**

```python
from pathlib import Path
from src.visualization.topology_visualizer import render_scion_topology_png, TopologyVisualizer

render_scion_topology_png(
    Path("evaluation/run_YYYYMMDD_HHMMSS/topology/scion_topology.json"),
    Path("topology_export.png"),
    dpi=200,
)

# Or the full dashboard + extras:
TopologyVisualizer().visualize_topology(
    Path("evaluation/run_YYYYMMDD_HHMMSS/topology/scion_topology.json"),
    Path("out/topology_dashboard.png"),
)
```

For a deeper file-by-file overview of the repository, see **`WALKTHROUGH.md`**.
