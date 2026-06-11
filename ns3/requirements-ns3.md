# Pinned NS-3 toolchain versions (ns3 branch)

NS-3, ns3-ai, and any QUIC module each target specific NS-3 versions; mismatches
are the single biggest integration risk on this branch. This is the source of
truth for the **known-good** combination used to build the `video_mpquic`
scenario (`ns3/examples/video-mpquic/`). Don't bump casually.

| Component   | Pinned version / commit | Notes |
|-------------|-------------------------|-------|
| NS-3        | `ns-3.42`               | Released 2024; builds cleanly against the pinned ns3-ai. |
| ns3-ai      | `b8c9858` (2025-01-23)  | `github.com/hust-diangroup/ns3-ai`. Provides the shared-memory bridge + `ns3ai_utils` (in `python_utils`). We use the **struct-based message interface**. |
| QUIC module | *(Phase 4 only)*        | `github.com/signetlabdei/quic` or chosen MPQUIC fork. Not needed for the abstracted-multipath MVP (Phases 1–3). |
| protobuf    | system `3.21.x`         | ns3-ai's bridge links protobuf; Ubuntu's `libprotobuf-dev` is fine. |
| pybind11    | `>=2.12` (pip 3.0.4 used) | Installed into the project venv (`uv pip install pybind11`); pass its cmake dir to `./ns3 configure`. |
| Boost       | headers only (1.8x+)    | Only header-only `boost/interprocess` is used — **no compiled boost libs**. See the patch note below. |
| g++ / CMake | g++ 13, CMake 3.28      | Ubuntu 24.04 defaults. C++17/20-capable. |
| Python      | 3.12 (venv)             | The `./ns3` wrapper and the pybind module both target the venv interpreter. Python 3.14 breaks the ns-3.42 `ns3` script (argparse). |

## Patches applied to ns3-ai (reproduced by `ns3/install_into_ns3.sh`)

1. **Boost: headers only.** ns3-ai's `contrib/ai/CMakeLists.txt` does
   `find_package(Boost REQUIRED COMPONENTS program_options)`, but `program_options`
   is never used — only header-only `boost/interprocess`. The install script
   rewrites it to `find_package(Boost REQUIRED)` so a header-only Boost (e.g.
   conda `libboost-headers`) satisfies the dependency without compiled libs.

## Build configuration that works (Ubuntu 24.04 / WSL2)

```bash
VENV_PY=/path/to/repo/.venv/bin/python
PYBIND_DIR=$($VENV_PY -c "import pybind11; print(pybind11.get_cmake_dir())")
BOOST_INC=$HOME/miniconda3/envs/ns3deps/include   # from: conda create -n ns3deps -c conda-forge libboost-headers

# Run ./ns3 with the venv python (3.12), not the system one:
$VENV_PY ns3 configure --enable-examples --disable-python --disable-werror -- \
  -DPython3_EXECUTABLE=$VENV_PY -DPython_EXECUTABLE=$VENV_PY \
  -Dpybind11_DIR=$PYBIND_DIR \
  -DBOOST_ROOT=$HOME/miniconda3/envs/ns3deps -DBoost_INCLUDE_DIR=$BOOST_INC -DBoost_NO_SYSTEM_PATHS=ON

$VENV_PY ns3 build ns3ai_video_mpquic
```

Memory note: on a 7–8 GB box cap the build (`cmake --build cmake-cache -j4 ...`);
a default `-j$(nproc)` link of NS-3 can OOM.

## Runtime constraint: one ns3-ai Experiment per Python process

ns3-ai allows only **one** shared-memory creator per Python process. A second
`Experiment` (a second `Ns3DataPlane` launch) in the same process fails with
`boost::interprocess_exception::library_error`. The `Ns3DataPlane` therefore
runs **one long-lived NS-3 process** and drives episode boundaries **in-band**
(`ACT_RESET`) over the bridge — the simulation is continuing across episodes.
Use a single `Ns3DataPlane` per process and `reset()` it each episode.

## Compatibility procedure (if bumping)

1. Pick the **ns3-ai** commit first; it constrains the NS-3 version.
2. Check out the matching **NS-3** tag.
3. Only at Phase 4, add the **QUIC** module and verify it builds against the same
   NS-3 tag (forks often break here).
4. Update the table above plus the protobuf/pybind11 versions ns3-ai pulled in.

## Why abstracted multipath first

The MVP (Phases 1–3) models N candidate paths as independent transport subflows
in `examples/video-mpquic/video_mpquic.cc` using stock NS-3 TCP, so it needs
**only NS-3 + ns3-ai** — no fragile MPQUIC module. True packet-level MPQUIC
scheduling is deferred to Phase 4, isolating the riskiest dependency until the
rest of the stack is proven.
