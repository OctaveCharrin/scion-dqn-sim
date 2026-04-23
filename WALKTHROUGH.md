# Codebase walkthrough

This document explains how the **SCION DQN simulation** repository is structured, how major pieces talk to each other, and where to look when you extend or fix things. It reflects the tree as of this writing; some paths are still evolving (see [Known gaps](#known-gaps)).

---

## What this project is trying to do

1. **Build** an AS-level network topology via **BRITE** (Java) and convert it to a SCION-style graph in JSON.
2. **Approximate SCION control-plane behavior** (beaconing, path discovery) so you get candidate paths between ASes.
3. **Simulate traffic and link conditions** over time.
4. **Train and evaluate** a **DQN** (and **baselines**) for **path selection** under **selective probing** (different probe costs, not probing every path).

The code is organized in layers: **topology generation** (`src/topology/`, BRITE), **simulation + RL** (`src/simulation/`, `src/beacon/`, `src/rl/`, `src/baselines/`), and **evaluation drivers** (`evaluation/*.py`) that form the main numbered pipeline.

---

## Big picture: evaluation-driven workflow

For RL training and paper-style experiments, the supported path is the **numbered scripts under `evaluation/`**, orchestrated by `run_full_evaluation.py`. Shared helpers live in `evaluation/_common.py` (run-directory resolution, pipeline subprocess runner, LNCS-style figure metadata).

```mermaid
flowchart TB
  subgraph eval_pipeline [evaluation pipeline]
    E01["01_generate_topology.py\nBRITE → .brite → SCION JSON"]
    E02["02_run_beaconing.py\npath_store.pkl"]
    E03["03_simulate_traffic.py\ntraffic + link states"]
    E04["04_train_dqn.py"]
    E05["05_evaluate_methods.py"]
    E06["06_generate_figures.py"]
    E01 --> E02 --> E03 --> E04 --> E05 --> E06
  end

  subgraph libs [shared libraries under src/]
    TOP["topology/*"]
    SIM["simulation/*"]
    RL["rl/*"]
    BASE["baselines/*"]
    PATH["path_services/*"]
    BEA["beacon/*"]
  end

  E01 -.-> TOP
  E02 -.-> BEA
  E02 -.-> SIM
  E04 -.-> RL
  E05 -.-> RL
  E05 -.-> BASE
```




| Entry point                         | Role                                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `evaluation/run_full_evaluation.py` | Runs `01`–`06` in order. Creates a timestamped `evaluation/run_*` directory by default, or pass `--run-dir PATH` to reuse an existing run. Uses `_common.run_script()` for each step. |


Evaluation uses **`topology/scion_topology.json`** (NetworkX node-link) plus run-scoped pickles (`path_store.pkl`, `link_states.pkl`, etc.). Older **pandas** `topology.pkl` + `link_table.parquet` flows still exist under `src/` for reuse (e.g. `BRITE2SCIONConverter.convert()`, traffic engine) but are not wired through a Typer CLI anymore.

---

## Repository layout


| Path                         | Purpose                                                                                                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `external/brite/`            | Git submodule: Boston University **BRITE** topology generator (Java). Built with `./setup_brite.sh` into `Java/Brite.jar`.                                                                                         |
| `configs/brite_templates/`   | Example `.conf` files matching BRITE’s numeric parser (used as reference; generator code builds equivalent text).                                                                                                  |
| `evaluation/`                | **End-to-end experiment drivers**: topology → beaconing → traffic → train → evaluate → figures. Each numbered script accepts `run_DIR` as `argv[1]` (or uses the latest `run_*` in the current directory). **`_common.py`** centralizes run-dir resolution, the orchestrator’s subprocess runner, and figure styling constants. |
| `tests/`                     | **Pytest suite** (`uv run pytest`). Covers topology config, traffic/link helpers, path aggregation, path store, baselines, DQN smoke, and `_common` helpers. Configured in `[tool.pytest.ini_options]` in `pyproject.toml`. |
| `src/topology/`              | BRITE **config generation** (`brite_cfg_gen.py`), **BRITE → SCION-ish graph** (`brite2scion_converter.py`).                                                                                                        |
| `src/beacon/`                | `**beacon_sim_v2.py`**: beacon simulation over `**topology.pkl`** (node/edge DataFrames).                                                                                                                          |
| `src/traffic/`               | `**traffic_engine.py**`: traffic matrix generation tied to topology pickles / memmaps.                                                                                                                             |
| `src/link_annotation/`       | `**capacity_delay_builder.py**`: annotate links from topology pickle.                                                                                                                                              |
| `src/path_services/`         | `**pathfinder_v2.py**`, `**pathprobe.py**`: path representation, probing metrics (used by harness / Gym-style RL envs).                                                                                              |
| `src/harness/`               | `**algo_harness.py**`: optional benchmark harness for path algorithms (pickle topology + memmaps); not used by the evaluation scripts.                                                                             |
| `src/baselines/`             | Six selector modules used by `05_evaluate_methods.py`: shortest, widest, lowest latency, ECMP, random, SCION default.                                                                                              |
| `src/rl/`                    | **`dqn_agent_enhanced.py`** (DQN used by the evaluation pipeline). **Gym-style** envs (`environment_*.py`), rewards (`reward_with_probing.py`), state (`state_enhanced.py`), `selective_probing_agent.py`—used by programmatic / research flows, **not** by `04`/`05` (those use `src.simulation.evaluation_env`). |
| `src/visualization/`         | Topology visualization helpers (optional; not required for the numbered evaluation steps).                                                                                                                       |
| `pyproject.toml` / `uv.lock` | **uv**-first packaging; `uv sync --extra dev` installs pytest and dev tools. `[tool.pytest.ini_options]` sets `testpaths = ["tests"]`.                                                                              |


---

## External tool: BRITE

- **Submodule**: `external/brite` (see `.gitmodules`).
- **Setup**: `./setup_brite.sh` checks Java, initializes the submodule, runs `make` in `Java/`, then `**jar cfe`** to build `Java/Brite.jar` (upstream Makefile only compiles classes).
- **Invocation contract**: `Main.Brite` expects **three** arguments: `config.conf`, **output path without `.brite` suffix**, and `**Java/seed_file`**. BRITE writes `**<stem>.brite`**.
- **Python side**: `BRITEConfigGenerator` writes a valid **numeric** BRITE config. **`run_brite()`** in `src/topology/brite_cfg_gen.py` is the single implementation that shells out to the JAR (config path, output stem without `.brite`, seed file). The fallback **`BRITERunner.run_parallel`** delegates to `run_brite()`. **`evaluation/01_generate_topology.py`** imports and calls `run_brite()` with the repo’s `external/brite` path.

---

## Evaluation pipeline (deep dive)

All steps share a directory like `evaluation/run_YYYYMMDD_HHMMSS/`. **`run_full_evaluation.py`** creates one by default (or you pass **`--run-dir PATH`**) and passes that path as `argv[1]` to each numbered script. Steps **`02`–`06`** call **`_common.resolve_run_dir()`** so they behave the same when run standalone: optional `argv[1]`, otherwise the lexicographically latest `run_*` in the current working directory (typically `evaluation/`).

### Step 1 — `01_generate_topology.py`

1. Creates **`topology/`** under the run directory. `**BRITEConfigGenerator**` writes **`topology/brite_config.conf`** (model codes such as AS Barabási–Albert / BA-2, `N`, bandwidth distribution, etc.). Node count can be overridden for smoke tests via env **`EVAL_BRITE_N_NODES`** (see script).
2. **`src.topology.brite_cfg_gen.run_brite()`** runs the JAR; output stem **`topology/topology`** → **`topology/topology.brite`**.
3. `**BRITE2SCIONConverter.convert_brite_file()**` reads the BRITE export, assigns **ISDs** (k-means on coordinates for multi-ISD; single ISD for small graphs), picks **core ASes**, adds **virtual edges** for connectivity / diversity, **classifies links**, adds **random PEER** edges for dense connectivity, and (when given **`plot_dir`**) saves **`step1_vanilla_brite.png`**, **`step2_scion_enhanced.png`**, **`step3_peering_enhanced.png`**. Returns a dict with:
  - `**graph`**: `networkx` graph (node attrs include `isd`, `x`, `y`; edges have `type`, `latency`, `bandwidth`),
  - `**isds`**: list of `{isd_id, member_ases}`,
  - `**core_ases`**: set of AS ids.
4. Writes **`topology/scion_topology.json`** (node-link graph + metadata) and **`topology/scion_topology.pkl`**.

**Downstream contract**: later steps load **`topology/scion_topology.json`** for dict/json usage (with a fallback to the legacy run-root path if present).

### Step 2 — `02_run_beaconing.py`

- Loads **`topology/scion_topology.json`** (or legacy **`scion_topology.json`** at run root), rebuilds a **NetworkX** graph, converts JSON → a temporary **`topology_beacon_input.pkl`** via **`src.simulation.json_topology_adapter.json_topology_to_beacon_pickle`** (DataFrame shape expected by the beacon simulator).
- Runs **`CorrectedBeaconSimulator`** from **`src.beacon.beacon_sim_v2`** (writes segment-like outputs under `beacon_output/`).
- Enumerates candidate paths with **`src.simulation.path_builder.build_paths_for_pair`**, picks a diverse **(src, dst)** pair, fills **`InMemoryPathStore`**, and saves **`path_store.pkl`** and **`selected_pair.json`**.

### Step 3 — `03_simulate_traffic.py`

- Reads topology, **selected pair**, `**path_store.pkl`**, generates 28 days of flow samples (hourly), builds `**traffic_flows.pkl`** and `**link_states.pkl**` for the same run directory.

### Step 4 — `04_train_dqn.py`

- Loads topology JSON, pair, path store, traffic, link states.
- Builds **`EvaluationPathSelectionEnv`** from **`src.simulation.evaluation_env`** (non-Gym adapter with `reset` / `step` / probing helpers aligned with the evaluation pipeline—not the Gymnasium `environment_*.py` stack).
- Trains **`EnhancedDQNAgent`** from **`src.rl.dqn_agent_enhanced`**; saves **`dqn_model.pth`** (including optimizer/scheduler state for reload in step 5).

### Step 5 — `05_evaluate_methods.py`

- Reloads **`EvaluationPathSelectionEnv`**, the trained checkpoint, and runs **DQN** vs the six baselines imported directly from **`src.baselines.*`** (`ShortestPathSelector`, `WidestPathSelector`, `LowestLatencySelector`, `ECMPSelector`, `RandomSelector`, `SCIONDefaultSelector`).

### Step 6 — `06_generate_figures.py`

- Reads **`evaluation_results.json`** and writes **PDF/PNG** figures. LNCS-style **`rcParams`**, column widths, method display names, and colors come from **`evaluation/_common.py`** (`apply_lncs_style`, `METHOD_DISPLAY_NAMES`, `METHOD_COLORS`, etc.) so they stay in sync with any future scripts.

---

## Library packages under `src/`

### Topology (`src/topology/`)


| Module                     | Responsibility                                                                                                                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `brite_cfg_gen.py`         | Valid BRITE `**.conf`** text; **`run_brite()`** (single JAR invocation); **`BRITERunner`** (parallel configs, delegates to `run_brite`). Legacy **`num_as`** kwargs map to **`n_nodes`** when **`n_nodes`** is not passed explicitly. |
| `brite2scion_converter.py` | Parse `**.brite`** topology text → **NetworkX**; ISD/core/virtual link logic; `**convert()`** → pickle with **nodes/edges DataFrames**; `**convert_brite_file()`** → dict for **evaluation JSON** path. |


**Interaction**: CLI `**generate`** uses `**convert()`**; evaluation `**01`** uses `**convert_brite_file()**`.

### Beacon (`src/beacon/beacon_sim_v2.py`)

- Loads `**topology.pkl**` with `**nodes` / `edges**` DataFrames.
- Simulates **core** and **intra-ISD** beacon phases, tracks **PCBs** and **interface IDs**, writes segment-like outputs under a run directory.

**Interaction**: Consumes **pickle** topology (`nodes` / `edges` DataFrames). The evaluation pipeline bridges **JSON → pickle** in step 2 via **`json_topology_to_beacon_pickle`**.

### Traffic (`src/traffic/traffic_engine.py`)

- `**TrafficEngine`**: time-slotted traffic generation from topology pickle paths.
- `**LinkMetricBuilder`**: derives per-link metrics from traffic memmaps + topology.

**Interaction**: Consumes `**topology.pkl`** and `**link_table.parquet`** in the CLI `simulate` flow.

### Link annotation (`src/link_annotation/capacity_delay_builder.py`)

- Consumes topology pickle, produces **annotated link table** (e.g. Parquet) for later stages.

### Path services (`src/path_services/`)

- `**pathprobe.py`**: models **path metrics** and probing cost.
- `**pathfinder_v2.py`**: `**SCIONPath`**, `**PathFinderV2**`—segment-aware path enumeration from topology + segment store + link table.

**Interaction**: `**algo_harness.py`** imports **`PathFinderV2`** and **`PathProbe`** from **`pathfinder_v2`** / **`pathprobe`**.

### Harness (`src/harness/algo_harness.py`)

- `**PathSelectionAlgorithm**` ABC; `**AlgorithmHarness**` runs Monte Carlo flows, records `**FlowResult**` statistics, can load algorithms by name from config.

**Interaction**: Benchmark layer on top of **path_services**; parallel to the **evaluation/** scripts but not identical.

### Baselines (`src/baselines/`)

- One module per policy (shortest, widest, lowest latency, ECMP, random, SCION default)—each exposes a `select_path(...)` used by **`05_evaluate_methods.py`**.

**Interaction**: **`05_evaluate_methods.py`** imports these classes directly (no separate registry module).

### RL (`src/rl/`)


| Area             | Files (representative)                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| **Agents (eval)**| `dqn_agent_enhanced.py` — used by **`04_train_dqn.py`** / **`05_evaluate_methods.py`**.           |
| **Agents (alt)** | `selective_probing_agent.py` — exported from `src.rl`; not wired into the numbered evaluation steps. |
| **Environments** | `environment_realistic.py` ← `environment_selective_probing.py` ← `environment_fixed_source.py` (Gymnasium). |
| **Rewards**      | `reward_with_probing.py`                                                                          |
| **State**        | `state_enhanced.py`                                                                               |


Gymnasium **API**: `reset` returns `(observation, info)` and `step` returns `(obs, reward, terminated, truncated, info)`. For `SCIONPathSelectionEnvFixedSource` (and subclasses), `source_as` / `dest_as` may be passed as `reset(options={...})` or as keywords `reset(source_as=..., dest_as=...)` for compatibility.

**Interaction chain for the numbered pipeline**: **topology JSON + path_store + link_states + traffic_flows** → **`EvaluationPathSelectionEnv`** → **`EnhancedDQNAgent`** → **`dqn_model.pth`**. The Gymnasium env stack is a **parallel** API for experiments that load pickle topologies + segment stores + link tables.

## Configuration and environment

- **Python**: Prefer **`uv sync --extra dev`** from repo root; use **`uv run python ...`** inside `evaluation/` for scripts (or **`uv run pytest`** from the repo root for tests).
- **Java**: Required for BRITE (`./setup_brite.sh`).
- **YAML**: Under `src/config/` for simulator / traffic settings (`sim.yml`, `traffic.yml`, etc.) if you extend those modules.

---

## Tests (`tests/`)

- **`uv run pytest`** discovers **`tests/`** via `pyproject.toml` (`[tool.pytest.ini_options]`).
- **`tests/conftest.py`** prepends the repo root to `sys.path` so `from src...` imports work regardless of cwd.
- Coverage today: BRITE config generation and `run_brite` error paths, traffic diurnal pattern and queueing delay, `PathProbe` static aggregators, `InMemoryPathStore`, all six baseline selectors (with stub path objects), `EnhancedDQNAgent` smoke, and **`evaluation/_common`** helpers.

---

## Known gaps (read this before a big refactor)

These are common tripping points when extending the simulator or the learning stack:

1. **Two topology shapes** — **`convert()`** (DataFrames + pickle) vs **`convert_brite_file()`** (NetworkX + JSON). The numbered pipeline is JSON-first; pickle/DataFrame flows remain for beacon input, traffic engine, and harness-style tooling.
2. **Evaluation imports** — Keep **`04_train_dqn.py`** / **`05_evaluate_methods.py`** aligned with **`src.simulation.evaluation_env`** and **`src.rl.dqn_agent_enhanced`**; grep before refactors.
3. **Two RL “worlds”** — Gymnasium envs under **`src/rl/environment_*.py`** vs the lightweight **`EvaluationPathSelectionEnv`** used by steps 4–5. Changes to state dimensions or reward definitions must stay consistent across both if you use both.

---

## How to improve it (practical order)

1. **Single topology model** — Define one schema (e.g. `TopologyBundle` with graph + isds + core + optional DataFrames) and functions **`to_json` / `from_pickle`** shared by `evaluation/` and pickle-based tools.
2. **Tests** — Extend **`tests/`** with integration cases: a tiny BRITE fixture + **`convert_brite_file`**, a minimal **`EvaluationPathSelectionEnv`** reset/step, optional end-to-end smoke with **`EVAL_BRITE_N_NODES`**.
3. **Observability** — **`_common.run_script`** still captures subprocess stdout/stderr; consider teeing to log files under `run_*` for long BRITE runs.

---

## Quick reference: artifact flow (evaluation / BRITE path)

```text
topology/brite_config.conf
topology/topology.brite ← BRITE JAR (+ seed_file)
topology/step1_vanilla_brite.png
topology/step2_scion_enhanced.png
topology/step3_peering_enhanced.png
topology/scion_topology.json  ← BRITE2SCIONConverter (+ peering in converter)
topology/scion_topology.pkl
path_store.pkl          ← intended: beaconing / path discovery
selected_pair.json
traffic_flows.pkl       ← traffic simulation
link_states.pkl
dqn_model.pth           ← training
evaluation_results.json ← baselines + DQN
figure*.pdf             ← plotting
```

---

## Suggested reading order for new contributors

1. `README.md` — install and high-level commands.
2. `evaluation/run_full_evaluation.py` — orchestration (`--run-dir` optional).
3. `evaluation/_common.py` — shared run-dir resolution, step runner, figure styling.
4. `evaluation/01_generate_topology.py` — BRITE glue + JSON contract.
5. `src/topology/brite2scion_converter.py` — **`convert_brite_file`** vs **`convert`**.
6. `src/simulation/evaluation_env.py` — what steps 4–5 actually use.
7. `src/beacon/beacon_sim_v2.py` — beacon simulation over DataFrames (fed from JSON in step 2).
8. `src/rl/environment_selective_probing.py` — Gym-side picture of path selection (optional deeper dive).
9. `src/harness/algo_harness.py` (optional) — benchmark harness; not required for the numbered `evaluation/` pipeline.

This should give you a mental model of **who calls whom**, **which file formats move between steps**, and **where the fragile boundaries** are when you extend the simulator or the learning stack.