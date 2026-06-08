# NS-3 data plane (ns3 branch)

This directory holds the **packet-level network** half of the ns3 branch: an NS-3
scenario for adaptive video over (abstracted) multipath QUIC, plus the glue that
exposes it to the Python RL agents via **ns3-ai** (shared memory).

It replaces, for this branch, the SCION analytical data plane and the BRITE-based
`setup_brite.sh` flow. The Python side (`src/ns3env/`, `src/rl/`) is unchanged
whether it talks to the mock data plane or to NS-3 — see the architecture diagram
in the project plan.

> **OS:** NS-3 and ns3-ai are Linux-only in practice. On Windows 11 use **WSL2
> Ubuntu**; everything here is portable to a native Linux/GPU server with no
> changes. Keep the repo on the Linux filesystem (`~/...`), **not** under
> `/mnt/c/...`, or NS-3 builds and training will be slow.

## Layout

```
ns3/
├── README.md                 # this file
├── requirements-ns3.md       # pinned versions (ns-3, ns3-ai, QUIC module)
└── scratch/
    └── video_mpquic.cc       # NS-3 scenario + ns3-ai message glue (skeleton)
```

The scenario is dropped into an NS-3 source tree's `scratch/` (or built as a small
module). It is not compiled on Windows; build it in WSL2.

## One-time setup (WSL2 Ubuntu)

```bash
# 0) System deps
sudo apt update
sudo apt install -y g++ cmake ninja-build python3 python3-pip git \
                    pkg-config libsqlite3-dev   # + protobuf for ns3-ai (see below)

# 1) Get NS-3 (pin a version — see requirements-ns3.md)
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev && git checkout <PINNED_NS3_TAG>

# 2) Add ns3-ai as a contrib module
git clone https://github.com/hust-diangroup/ns3-ai.git contrib/ai
#   then follow contrib/ai/install.md (it builds a pybind11 + protobuf bridge and
#   installs the `ns3ai_gym_env` / msg-interface Python packages into your venv)

# 3) (Phase 4, optional) add a QUIC module for true MPQUIC
#    e.g. signetlabdei/quic — pin the commit in requirements-ns3.md

# 4) Place the scenario and configure/build
cp /path/to/repo/ns3/scratch/video_mpquic.cc scratch/
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

## Python environment (WSL2)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
cd /path/to/repo
uv sync --extra dev                                # torch, numpy, pytest, ...
#   then install the ns3-ai Python bindings produced in step 2 above into .venv
```

The pure-Python parts (env, reward, baselines, agents, training on the
**mock** data plane) run **without NS-3** on any OS — that is what CI / the
`tests/test_video_*.py` suite exercises. NS-3 is only needed for the real
`Ns3DataPlane`.

## How the two halves connect

Decision epoch = **one video segment download**. Per epoch:

1. C++ fills a per-path observation struct (throughput EWMA, RTT, loss) and blocks.
2. Python (`Ns3DataPlane`) reads it, the env assembles
   `{"global": (G,), "paths": (N, P)}`, the agent picks a path (Phase 2+: also
   rate / bitrate), and Python writes the action struct back.
3. C++ runs the simulation until the chosen segment is delivered, records the
   realized goodput/RTT/loss, and loops.

`Ns3DataPlane.download_segment()` and `current_path_stats()` (in
`src/ns3env/dataplane.py`) are the Python ends of this loop; the struct fields in
`scratch/video_mpquic.cc` mirror `PathStats` / `DownloadResult`.

## Smoke checks

```bash
# Bridge sanity (before any project code): run a stock ns3-ai example end-to-end.
# Then, once video_mpquic is built, the Ns3DataPlane parity check:
uv run python -m pytest tests/test_video_env.py -q        # mock backend (works anywhere)
# Ns3DataPlane integration test is added in Phase 1 once the scenario builds.
```
