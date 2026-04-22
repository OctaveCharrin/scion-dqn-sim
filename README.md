# SCION Network Simulator for DQN-based Path Selection

A simulator for SCION networks with Deep Q-Network (DQN) based path selection optimization.

Note: This is a work in progress and some components are not yet fully implemented.

## Features

- **BRITE-based Topology Generation**: Create realistic AS-level SCION topologies
- **Full SCION Control Plane Simulation**: Beaconing, path discovery, and segment registration
- **Traffic Simulation**: Gravity and uniform traffic models with diurnal patterns
- **Deep Reinforcement Learning**: DQN agent for intelligent path selection
- **Performance Metrics**: Metrics collection and visualization
- **Selective Probing**: DQN-based intelligent path probing to reduce overhead
- **Baseline Comparisons**: Compare DQN against shortest path, widest path, lowest latency, ECMP, random, and SCION default selectors

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/netsys-lab/scion-dqn-sim.git
cd my-scion-dqn-sim
```

Or if you already cloned without submodules:
```bash
git submodule update --init --recursive
```

2. Set up BRITE topology generator:
```bash
./setup_brite.sh
```

3. Install Python tooling with [uv](https://docs.astral.sh/uv/) and sync dependencies (creates `.venv` and installs the package in editable mode):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or use your OS package manager
uv sync --extra dev
```

Use `uv run python ...` or `source .venv/bin/activate` so scripts use the project environment.

The legacy `requirements.txt` is kept in sync with `pyproject.toml` for reference; prefer `uv sync` for installs.

## Quick Start

### Running the Complete Evaluation Pipeline

The evaluation pipeline compares DQN-based path selection with baseline methods (shortest path, widest path, lowest latency, ECMP, random, SCION default) on SCION networks.

**Run the complete evaluation:**

```bash
cd evaluation
uv run python run_full_evaluation.py
```

or

```bash
cd evaluation
uv run python run_full_evaluation_2.py
```

This will execute the complete 6-step pipeline:
1. **Generate Topology**: Creates a dense SCION topology using BRITE
2. **Run Beaconing**: Simulates SCION beaconing to discover paths between ASes
3. **Simulate Traffic**: Generates 28 days of traffic with diurnal and weekly patterns
4. **Train DQN**: Trains the DQN agent on the first 14 days of traffic
5. **Evaluate Methods**: Compares all methods on the last 14 days of traffic
6. **Generate Figures**: Creates visualization figures comparing performance

**Key Features:**
- **Selective Probing**: DQN only probes selected paths while baseline methods must probe all paths
- **Differentiated Probe Costs**: Latency probes (10ms) vs bandwidth probes (100ms)
- **Realistic Traffic**: Diurnal and weekly patterns with gravity and uniform models
- **Fair Comparison**: All methods evaluated on the same traffic flows

**Outputs:**
All results are saved in a timestamped run directory (e.g., `run_20250805_071054/`):
- `scion_topology.json`: Network topology
- `selected_pair.json`: Source-destination AS pair used for evaluation
- `dqn_model.pth`: Trained DQN model
- `evaluation_results.json`: Performance comparison metrics
- `figure1_probe_overhead.pdf`: Probe overhead comparison
- `figure2_path_reward.pdf`: Path reward distribution
- `figure3_probe_breakdown.pdf`: Probe type breakdown

### Running Individual Steps

You can also run individual steps manually:

```bash
cd evaluation

# Create a run directory
mkdir -p run_YYYYMMDD_HHMMSS

# Step 1: Generate topology
uv run python 01_generate_topology.py run_YYYYMMDD_HHMMSS

# Step 2: Run beaconing
uv run python 02_run_beaconing.py run_YYYYMMDD_HHMMSS

# Step 3: Simulate traffic
uv run python 03_simulate_traffic.py run_YYYYMMDD_HHMMSS

# Step 4: Train DQN
uv run python 04_train_dqn.py run_YYYYMMDD_HHMMSS

# Step 5: Evaluate methods
uv run python 05_evaluate_methods.py run_YYYYMMDD_HHMMSS

# Step 6: Generate figures
uv run python 06_generate_figures.py run_YYYYMMDD_HHMMSS
```

### Evaluation Metrics

The evaluation compares the following metrics across all path selection methods:

- **Reward**: Composite metric combining goodput, latency, and loss rate
- **Latency**: Path latency in milliseconds (mean, p50, p95)
- **Bandwidth**: Available bandwidth in Mbps
- **Probe Overhead**: Number and time cost of latency/bandwidth probes
  - Latency probes: 10ms base cost + 0.5ms per hop
  - Bandwidth probes: 100ms base cost + 20ms per hop
- **Selection Time**: Time taken to select a path
- **Probe Reduction**: Percentage reduction in probe overhead compared to baseline methods

The DQN agent uses **selective probing** - it only probes the path it intends to select, while baseline methods must probe all available paths before making a decision. This results in significant probe overhead reduction while maintaining competitive performance.

### Programmatic Usage

You can also use the simulator programmatically:

```python
from src.topology import SCIONTopologyGenerator

generator = SCIONTopologyGenerator()
result = generator.generate(
    num_ases=100,
    num_isds=3,
    topology_type='medium',
    seed=42
)
```

