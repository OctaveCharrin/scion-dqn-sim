# Pinned NS-3 toolchain versions (ns3 branch)

NS-3, ns3-ai, and any QUIC module each target specific NS-3 versions; mismatches
are the single biggest integration risk on this branch. Pin them here and treat
this file as the source of truth. Fill the `<TBD>` fields when the WSL2 setup is
first done, then do not bump casually.

| Component   | Pinned version / commit | Notes |
|-------------|-------------------------|-------|
| NS-3        | `<TBD: e.g. ns-3.42>`   | Choose the newest version ns3-ai officially supports. |
| ns3-ai      | `<TBD: commit hash>`    | `github.com/hust-diangroup/ns3-ai`. Provides the shared-memory bridge + `ns3ai_gym_env` / msg-interface Python packages. |
| QUIC module | `<TBD>` (Phase 4 only)  | `github.com/signetlabdei/quic` or chosen MPQUIC fork. Not needed for the abstracted-multipath MVP (Phases 1–3). |
| protobuf    | match ns3-ai's requirement | ns3-ai's bridge depends on a specific protobuf range. |
| g++ / CMake | distro defaults on Ubuntu 22.04+ | NS-3 needs C++17/20-capable g++ and CMake ≥ 3.13. |

## Compatibility procedure

1. Pick the **ns3-ai** release first; read its README for the exact NS-3
   version(s) it supports — that constrains everything else.
2. Check out that **NS-3** tag.
3. Only at Phase 4, add the **QUIC** module and verify it builds against the same
   NS-3 tag (this is where forks often break — be ready to patch or pick a
   different fork).
4. Record the working combination in the table above, plus the protobuf and
   pybind11 versions that ns3-ai pulled in.

## Why abstracted multipath first

The MVP (Phases 1–3) models N candidate paths as independent transport subflows
in `scratch/video_mpquic.cc` using stock NS-3 transports (TCP/QUIC single-path),
so it needs **only NS-3 + ns3-ai** — no fragile MPQUIC module. True packet-level
MPQUIC scheduling is deferred to Phase 4, isolating the riskiest dependency until
the rest of the stack is proven.
