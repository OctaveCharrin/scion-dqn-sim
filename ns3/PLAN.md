> **Status (2026-06-11).** Phase 0 and Phase 1 are **done and verified**: the
> ns3-ai bridge works, `Ns3DataPlane` + `MockTraceDataPlane` + `Ns3VideoMpquicEnv`
> are implemented, the path-scoring agent trains end-to-end on both backends, and
> the suite is green (`tests/test_video_*.py` everywhere; `tests/test_ns3_dataplane_*`
> on the built WSL2 tree). One path note: the NS-3 scenario landed at
> `ns3/examples/video-mpquic/` (built under the ns-3 tree's `contrib/ai/examples/`),
> **not** `ns3/scratch/` as the original plan text below says. Phases 2–4 are open.
>
> **Phase 1 milestone validated** via `src/rl/video_eval_compare.py` (train DQN +
> compare vs all `MULTIPATH_SELECTORS` on a shared data plane). On the `crossover`
> mock scenario (best path rotates over time) the DQN reaches reward ~0.844,
> decisively beating round_robin (0.295), static_path0 (0.592) and single_best
> (0.614), and *matching* the reactive `max_throughput` oracle (0.844). It can't
> exceed reactive greedy because the abstract mock is memoryless per step — that
> headroom is what Phase 2 (playback buffer / rebuffering / switching cost) adds.
> Run: `uv run python -m src.rl.video_eval_compare --scenario crossover`.
>
> **Phase 2 in progress (Option A, two steps).** Step 1 done & validated:
> single-path adaptive bitrate (ABR) with a client playback buffer and a
> **VMAF-based** QoE reward (perceptual quality − rebuffer − VMAF switching;
> `src/ns3env/abr.py`, `abr_env.py`). The buffer adds the delayed-consequence
> structure RL needs: on the `varying` mock the DQN (reward 0.641) **beats**
> buffer-based/BOLA (0.598), rate-based (0.449) and fixed policies — it manages
> the quality↔rebuffer↔smoothness tradeoff greedy heuristics botch. Quality uses
> a concave log VMAF(bitrate) curve (Netflix "VMAF: The Journey Continues"),
> swappable for measured per-content VMAF. Run:
> `uv run python -m src.rl.video_abr_train --scenario varying`.
>
> Step 2 done & validated: **joint (path × bitrate)** multipath ABR
> (`src/ns3env/abr_joint_env.py`) — each (path, bitrate) pair is one scoring
> candidate, so the same agent jointly picks path and quality (no agent change).
> On a capacity-constrained 3-path `crossover` scenario the DQN (reward 0.835,
> VMAF 86.9, 1.9 s/ep rebuffer) **beats** best-path-rate (0.795), best-path-buffer
> (0.735) and best-path-max (0.728, which hits VMAF 92 but 35.6 s/ep rebuffer). It
> reaches higher quality than the rate/buffer heuristics while keeping stalls low.
> Run: `uv run python -m src.rl.video_abr_train --mode joint`. **Phase 2 complete.**
> Remaining (Phase 3): wire the joint ABR onto the real NS-3 backend at scale, add
> MPC/BOLA-proper baselines, and the numbered evaluation pipeline + figures.

# Plan: Migrate to NS-3 + RL-controlled adaptive video over (abstracted) multipath QUIC

## Context

The current repo (`main`) is a **SCION AS-level path-selection** simulator: an analytical,
hourly-aggregated traffic model (`src/simulation/`), BRITE topologies + SCION beaconing
(`src/topology/`, `src/beacon/`), and a mature family of PyTorch DQN agents that pick a
path index (`src/rl/`). The reward trades off goodput vs a link-trust score, with
multi-objective conditioning via reward-weight FiLM (`dqn_agent_scoring_conditional.py`).

The new direction (this `ns3` branch) is different enough that it is **not really a
"migration of the simulator" — it is reusing the RL brain on a new body**:

- Replace the SCION/analytical data plane with a **packet-level NS-3 data plane**.
- Insert a **decision/abstraction layer between a video application and the network** that an
  RL agent controls: which path to use, sending/pacing rate, and chunk/bitrate.
- Target **adaptive video streaming over multipath QUIC**, optimizing QoE.

**What carries over (reused, not rewritten):**
- All of `src/rl/` — agents, replay buffers, dueling/Double-DQN, PER, and the weight-FiLM
  conditional agent. The variable-cardinality **path-scoring** agent
  (`DuelingPathScoringDQN` / `EnhancedPathScoringDQNAgent` in
  `src/rl/dqn_agent_scoring_enhanced.py`) fits "choose among N paths" almost unchanged.
- The **observation contract** `{"global": (G,), "paths": (N, P)}` and `RewardWeights` /
  `reward_profiles.py` (multi-objective conditioning) from `src/simulation/evaluation_env.py`.
- The **pipeline conventions**: numbered steps, timestamped run dirs, subprocess orchestration
  (`src/pipeline/run_dirs.py`), pytest patterns.

**What is dropped / not used on this branch:** BRITE topology generation, SCION beaconing,
`path_store`, the hourly `link_hourly_v1` traffic model. (Left intact on `main`.)

**Settled decisions (from clarification):**
1. Data plane + RL bridge: **NS-3 + ns3-ai** (shared-memory; faster & better maintained than ns3-gym; keeps the PyTorch agent in Python).
2. MPQUIC fidelity: **abstracted multipath first** — model N candidate paths as independent transport subflows; the agent picks path(s) + rate. True MPQUIC packet scheduling is a later stretch.
3. Action space: **phased** — path selection first (reuse agents), then sending rate, then chunk/bitrate (full ABR).
4. Environment: **WSL2 Ubuntu now**, portable to a native Linux/GPU server later (no porting work).

## Target architecture

```
            ┌──────────────────────── Python (reuse src/rl) ─────────────────────────┐
            │  Ns3VideoMpquicEnv  →  observe() {"global":(G,), "paths":(N,P)}          │
            │       │ step(action)           agent.act() / agent.replay()              │
            │       ▼                         (EnhancedPathScoringDQNAgent, etc.)      │
            │  DataPlane (abstract)                                                    │
            │     ├── Ns3DataPlane  ── ns3-ai shared memory ──┐                        │
            │     └── MockTraceDataPlane (pure Python, trace-driven, for tests/CI)     │
            └─────────────────────────────────────────────────│──────────────────────┘
                                                               ▼
            ┌──────────────────────── NS-3 C++ scenario (thin) ───────────────────────┐
            │  Multi-homed client ── N bottleneck paths (P2P + queue + cross-traffic)  │
            │  N transport subflows (ns-3 QUIC module or TCP)  +  DASH video app       │
            │  Ns3AiGymEnv glue: fill obs struct → block → read action → continue      │
            └─────────────────────────────────────────────────────────────────────────┘
```

Key design choice: a **`DataPlane` abstraction** with two backends. `Ns3DataPlane` drives the
real NS-3 sim via ns3-ai; `MockTraceDataPlane` replays bandwidth/RTT traces in pure Python.
This keeps the project's fast-iteration ethos: the Python env, agents, reward, and baselines
are fully unit-testable on Windows **without compiling NS-3**, and RL development is decoupled
from the C++ integration risk.

## Topology / scenario (default)

Classic MPQUIC-video setting: one **multi-homed client** with **N access paths** (e.g., 2–3
links emulating Wi-Fi / LTE / wired) to a video server. Each path = `PointToPointChannel`
with configurable bandwidth + delay + a queue/`TrafficControl` for loss, plus an `OnOff`
cross-traffic app to make congestion time-varying. N is configurable; the path-scoring agent
already handles variable N. Decision epoch = **per video segment** (natural ABR cadence),
not hourly.

## Work phases

### Phase 0 — Environment & bridge de-risking (do this first, in isolation)
- WSL2 Ubuntu: install NS-3 (pin a version known to work with ns3-ai, e.g. ns-3.40/3.41),
  `uv`, PyTorch. Add NS-3/ns3-ai setup notes to a new `ns3/README.md` (replacing the
  bash-only `setup_brite.sh` role for this branch).
- Build and run an **ns3-ai example end-to-end** to prove the Python↔C++ shared-memory loop
  before writing any project code. This is the single biggest risk; validate it early.
- Pin versions in a new `ns3/requirements-ns3.md` (ns-3 version, ns3-ai commit, QUIC module
  commit). Add `ns3-ai` to the Python env.

### Phase 1 — Abstracted multipath + path-selection agent (first working closed loop)
- **NS-3 scenario** (`ns3/scratch/video_mpquic.cc` or a small module): N-path topology, N
  independent transport subflows, a simple segment "downloader" app, cross-traffic. Expose
  per-path stats (EWMA throughput, RTT, loss, in-flight) + global stats (buffer, time) through
  the ns3-ai obs struct; accept a discrete path action.
- **Python env** `src/ns3env/video_env.py` → `Ns3VideoMpquicEnv` implementing the existing
  gym-like contract (`reset()`, `step(action) -> (obs, reward, done, info)`,
  `observe()` returning `{"global", "paths"}`). Per-path feature vector mirrors the existing
  7-D layout so `DuelingPathScoringDQN` is reusable unchanged.
- **DataPlane backends**: `src/ns3env/dataplane.py` with `Ns3DataPlane` (ns3-ai) and
  `MockTraceDataPlane`.
- **Training entrypoint** `src/rl/video_mpquic_train.py` — adapt the loop in
  `src/rl/path_selection_train.py::train_scoring_dqn` to the new env (reuse the agent,
  `EnhancedDQNConfig`, replay/epsilon logic verbatim).
- **Reward** (`src/ns3env/reward.py`): start with throughput/goodput so we can confirm the
  loop learns, expressed via the existing `RewardWeights` so it stays compatible with later
  QoE/conditional work.
- **Baselines** (`src/baselines/` parallel additions): round-robin, min-RTT, single-best-path,
  redundant — same `select_path(...)->int` interface the repo already uses.
- **Milestone:** agent beats round-robin/single-path on a deliberately asymmetric, time-varying
  scenario, in both the mock and NS-3 data planes.

### Phase 2 — Real DASH video app + QoE reward + sending-rate action
- Add a **DASH-style video model**: standard bitrate ladder (Pensieve-style, e.g.
  {0.3, 0.75, 1.2, 1.85, 2.85, 4.3} Mbps), fixed-duration segments, client playback buffer,
  rebuffering accounting. Driven from NS-3 (and mirrored in the mock for tests).
- **QoE reward**: `r = bitrate − μ·rebuffer − λ·|Δbitrate|`, expressed inside the
  `RewardWeights` framework so `reward_profiles.py` still applies.
- **Action space += sending/pacing rate** per chosen path (rate split for multipath).

### Phase 3 — Full ABR + multi-objective + evaluation
- **Action space += chunk/bitrate** (full ABR: path + rate + bitrate). For the joint space,
  either factorize into the scoring head per candidate (path×bitrate) or add a small bitrate
  head; prefer reusing the scoring formulation over a fixed flat action space.
- **Multi-objective conditioning**: reuse `ConditionalPathScoringDQNAgent` (weight-FiLM) to
  trade off quality vs smoothness vs rebuffering at inference time.
- **ABR baselines**: rate-based, buffer-based (BOLA), MPC — for comparison.
- **Pipeline + figures**: new numbered steps under `evaluation/` mirroring the existing
  convention (`01_generate_scenario` → `02_build_ns3` → `03_train` → `04_evaluate` →
  `05_figures`), orchestrated via the existing `src/pipeline/run_dirs.py` pattern; QoE/throughput
  comparison plots via the `src/pipeline/figures.py` styling.

### Phase 4 (stretch) — True MPQUIC
- Swap the abstracted N-subflow scheduler for a real MPQUIC ns-3 module and let the agent act
  as / tune the packet scheduler. Only attempt once Phases 1–3 are solid (version lock-in and
  module maturity are the main risks).

## Critical files

**Reuse as-is:** `src/rl/dqn_agent_scoring_enhanced.py`,
`src/rl/dqn_agent_scoring_conditional.py`, `src/rl/dqn_agent_enhanced.py` (`EnhancedDQNConfig`),
`src/rl/reward_profiles.py`, the `RewardWeights` + `observe_scoring` contract in
`src/simulation/evaluation_env.py`, `src/pipeline/run_dirs.py`, `src/pipeline/figures.py`.

**New (this branch):**
- `ns3/` — NS-3 scenario C++ (`scratch/video_mpquic.cc`), build + setup docs (`ns3/README.md`).
- `src/ns3env/dataplane.py` — `DataPlane` abstract + `Ns3DataPlane` (ns3-ai) + `MockTraceDataPlane`.
- `src/ns3env/video_env.py` — `Ns3VideoMpquicEnv` (gym-like contract).
- `src/ns3env/reward.py` — QoE reward in the `RewardWeights` framework.
- `src/rl/video_mpquic_train.py` — training entrypoint (adapted from `path_selection_train.py`).
- `src/baselines/` — round-robin / min-RTT / single-best / redundant selectors (+ ABR baselines in Phase 3).
- `evaluation/` — new numbered pipeline scripts for the video/MPQUIC run.
- `tests/test_video_env.py`, `tests/test_video_baselines.py`, `tests/test_video_reward.py`.

**Adapt dependencies:** add `ns3-ai` to the Python env via `uv`; note `gymnasium` may be added
if we want a strict Gym API (current env is custom but Gym-like).

## Verification

- **Phase 0:** the stock ns3-ai example runs the full Python↔C++ loop in WSL2 (proves the bridge).
- **Mock-backed unit tests (run on Windows, no NS-3):** `pytest tests/test_video_env.py`
  checks env contract (`reset`/`step` shapes, reward bounds, variable N); reward and baseline
  tests check correctness; a short training smoke run on `MockTraceDataPlane` shows reward
  increasing on an asymmetric scenario.
- **Phase 1 integration (WSL2):** run training against `Ns3DataPlane`; confirm the agent beats
  round-robin and single-best-path on a time-varying scenario; confirm parity of learned
  behavior between mock and NS-3 backends.
- **Phase 2–3:** end-to-end pipeline produces `evaluation/run_*/` with QoE results JSON and
  comparison figures; DQN vs ABR baselines (BOLA/MPC) on held-out network traces; conditional
  agent shows the expected quality/smoothness trade-off as weights vary.
- Keep `uv run black/flake8` and `uv run python -m pytest` green throughout (Windows-runnable
  via the mock backend).
