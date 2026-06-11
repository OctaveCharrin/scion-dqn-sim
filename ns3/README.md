# NS-3 data plane (ns3 branch)

This directory holds the **packet-level network** half of the ns3 branch: an NS-3
scenario for adaptive video over (abstracted) multipath QUIC, plus the glue that
exposes it to the Python RL agents via **ns3-ai** (shared memory).

It replaces, for this branch, the SCION analytical data plane and the BRITE-based
`setup_brite.sh` flow. The Python side (`src/ns3env/`, `src/rl/`) is unchanged
whether it talks to the mock data plane or to NS-3 — both implement the same
`DataPlane` interface.

> **OS:** NS-3 and ns3-ai are Linux-only in practice. On Windows 11 use **WSL2
> Ubuntu**; everything here is portable to a native Linux/GPU server with no
> changes. Keep the repo on the Linux filesystem (`~/...`), **not** under
> `/mnt/c/...`, or NS-3 builds and training will be slow.

## Layout

```
ns3/
├── README.md                    # this file
├── requirements-ns3.md          # pinned versions + the known-good build config
├── install_into_ns3.sh          # copy the example into an ns-3 tree + apply patches
└── examples/
    └── video-mpquic/            # the ns3-ai example (struct-based message interface)
        ├── video_mpquic.h       # EnvStruct / ActStruct — the shared-memory wire format
        ├── video_mpquic.cc      # NS-3 scenario + per-segment decision loop
        ├── video_mpquic_py.cc   # pybind11 binding (module: ns3ai_video_mpquic_py)
        └── CMakeLists.txt       # build_lib_example + pybind11_add_module
```

The example is built **inside** an ns-3 source tree under
`contrib/ai/examples/video-mpquic/` (the canonical copy lives here in the repo;
`install_into_ns3.sh` syncs it into the tree). It is not compiled on Windows.

## One-time setup (WSL2 Ubuntu 24.04)

See `requirements-ns3.md` for the **pinned versions** and the exact, known-good
`./ns3 configure` line. In short:

```bash
# 0) System deps already present on a typical dev box: g++, cmake, ninja, git,
#    libprotobuf-dev + protoc, libsqlite3-dev. Boost + pybind11 are obtained
#    WITHOUT sudo:
uv pip install pybind11                                   # into the project venv
conda create -y -n ns3deps -c conda-forge libboost-headers  # header-only boost

# 1) NS-3 (pinned tag) + ns3-ai (pinned commit) as a contrib module
git clone --depth 1 --branch ns-3.42 https://gitlab.com/nsnam/ns-3-dev.git ~/ns-3-dev
git clone https://github.com/hust-diangroup/ns3-ai.git ~/ns-3-dev/contrib/ai

# 2) Drop in the video-mpquic example and apply the ns3-ai patches
/path/to/repo/ns3/install_into_ns3.sh ~/ns-3-dev

# 3) Configure (use the venv python — the ns-3.42 ./ns3 script breaks on py3.14)
#    and build only what we need. See requirements-ns3.md for the full flags.
cd ~/ns-3-dev
$REPO/.venv/bin/python ns3 configure --enable-examples --disable-python --disable-werror -- \
  -DPython3_EXECUTABLE=$REPO/.venv/bin/python -DPython_EXECUTABLE=$REPO/.venv/bin/python \
  -Dpybind11_DIR=$($REPO/.venv/bin/python -c 'import pybind11;print(pybind11.get_cmake_dir())') \
  -DBOOST_ROOT=$HOME/miniconda3/envs/ns3deps \
  -DBoost_INCLUDE_DIR=$HOME/miniconda3/envs/ns3deps/include -DBoost_NO_SYSTEM_PATHS=ON
cmake --build cmake-cache -j4 --target ns3ai_video_mpquic   # -j4: avoid OOM on ~8GB

# 4) Python bridge helper (ns3ai_utils) into the venv
uv pip install -e ~/ns-3-dev/contrib/ai/python_utils
```

## Python environment

The pure-Python parts (env, reward, baselines, agents, training on the **mock**
data plane) run **without NS-3** on any OS — that is what CI / the
`tests/test_video_*.py` suite exercises. NS-3 is only needed for the real
`Ns3DataPlane` and the `tests/test_ns3_dataplane_integration.py` suite (which
skips when the scenario isn't built).

## How the two halves connect

Decision epoch = **one video segment download**. Each decision is one
shared-memory round-trip (C++ leads with a send):

1. C++ fills an `EnvStruct` (per-path throughput EWMA / RTT / loss + the realized
   result of the previous segment) and sends it; it then blocks for an action.
2. Python (`Ns3DataPlane`) reads it, the env assembles
   `{"global": (G,), "paths": (N, P)}`, the agent picks a path, and Python sends
   an `ActStruct` back.
3. C++ delivers that segment over the chosen subflow (running the simulation
   until the bytes arrive), records the realized goodput/RTT/loss, and loops.

`Ns3DataPlane.current_path_stats()` / `download_segment()` (in
`src/ns3env/dataplane.py`) are the Python ends of this loop; the struct fields in
`video_mpquic.h` mirror `PathStats` / `DownloadResult`.

### One process, many episodes (important)

ns3-ai allows only **one** shared-memory creator per Python process. So
`Ns3DataPlane` launches a **single long-lived NS-3 process** and drives episode
boundaries **in-band**: `reset()` sends `ACT_RESET`, which zeroes the episode
counters while the simulation keeps running (warm subflows, evolving congestion).
Use one `Ns3DataPlane` per process and `reset()` it each episode; do not create a
second `Ns3DataPlane` in the same process. `close()` sends `ACT_TERMINATE` and
frees the shared memory.

## Running

```bash
export NS3_DIR=~/ns-3-dev    # or pass ns3_dir= to Ns3DataPlane

# Standalone C++ sanity check (no Python bridge): one round-robin episode.
~/ns-3-dev/build/contrib/ai/examples/video-mpquic/ns3.42-ns3ai_video_mpquic-default \
    --selftest=1 --segments=8 --segmentBytes=300000

# Integration tests (skip automatically if the scenario isn't built):
uv run python -m pytest tests/test_ns3_dataplane_integration.py -q

# Mock backend (works anywhere, no NS-3):
uv run python -m pytest tests/test_video_env.py -q
```

Programmatic use:

```python
from src.ns3env import Ns3DataPlane, Ns3Config, Ns3VideoMpquicEnv

dp = Ns3DataPlane(config=Ns3Config(episode_segments=48, segment_bytes=500_000, seed=1))
env = Ns3VideoMpquicEnv(dp)        # same env as the mock backend
obs = env.reset()
obs, reward, done, info = env.step(action)
...
dp.close()
```
