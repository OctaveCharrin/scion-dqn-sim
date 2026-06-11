"""
NS-3 / multipath-QUIC video environment for RL-controlled adaptive streaming.

This package is the ``ns3`` branch's replacement for the SCION analytical data
plane. It exposes a Gym-like environment (:class:`~src.ns3env.video_env.Ns3VideoMpquicEnv`)
backed by a pluggable :class:`~src.ns3env.dataplane.DataPlane`:

* :class:`~src.ns3env.dataplane.MockTraceDataPlane` -- pure-Python, trace-driven;
  runs anywhere (Windows/CI) with no NS-3 dependency. Used for fast RL iteration
  and unit tests.
* :class:`~src.ns3env.dataplane.Ns3DataPlane` -- drives a real NS-3 packet-level
  scenario over the ns3-ai shared-memory bridge (Linux/WSL2 only).

The environment reuses the existing path-scoring agents in :mod:`src.rl`
unchanged via the observation contract ``{"global": (G,), "paths": (N, P)}``.
"""

from src.ns3env.dataplane import (
    DataPlane,
    DownloadResult,
    MockTraceConfig,
    MockTraceDataPlane,
    Ns3Config,
    Ns3DataPlane,
    PathStats,
)
from src.ns3env.reward import RewardWeights, compute_reward
from src.ns3env.video_env import (
    GLOBAL_DIM,
    PATH_FEATURE_DIM,
    Ns3VideoMpquicEnv,
)

__all__ = [
    "DataPlane",
    "DownloadResult",
    "PathStats",
    "MockTraceConfig",
    "MockTraceDataPlane",
    "Ns3Config",
    "Ns3DataPlane",
    "RewardWeights",
    "compute_reward",
    "Ns3VideoMpquicEnv",
    "GLOBAL_DIM",
    "PATH_FEATURE_DIM",
]
