"""
Gym-like environment for RL-controlled adaptive video over multipath QUIC.

This is the ``ns3`` branch's analogue of ``EvaluationPathSelectionEnv``. It keeps
the same observation contract -- ``{"global": (G,), "paths": (N, P)}`` -- so the
existing path-scoring agents in :mod:`src.rl` (``EnhancedPathScoringDQNAgent``,
``ConditionalPathScoringDQNAgent``) are reusable unchanged.

Phase 1 action space: a discrete path index (which subflow to fetch the next
segment on). Phases 2-3 extend ``step`` to also accept a sending rate and a
chunk/bitrate (full ABR) -- the observation/reward shape is designed to grow
without breaking the agent interface.

The environment is backend-agnostic: it talks only to a :class:`DataPlane`
(``MockTraceDataPlane`` for tests/CI, ``Ns3DataPlane`` for real NS-3 runs).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.ns3env.dataplane import DataPlane, PathStats
from src.ns3env.reward import RTT_TRUST_REF_MS, RewardWeights, compute_reward

# Global context: [progress, last_goodput_norm, last_rtt_norm, mean_throughput_norm].
GLOBAL_DIM = 4

# Per-path: [throughput_norm, rtt_norm, loss, throughput_ratio, last_chosen_flag].
PATH_FEATURE_DIM = 5

Observation = Dict[str, np.ndarray]

_DEFAULT_SEGMENT_BYTES = 500_000


class Ns3VideoMpquicEnv:
    """Per-segment path-selection environment over a pluggable data plane."""

    def __init__(
        self,
        dataplane: DataPlane,
        *,
        reward_weights: Optional[RewardWeights] = None,
        segment_bytes: Optional[int] = None,
    ):
        self.dataplane = dataplane
        self.reward_weights = reward_weights or RewardWeights()
        if segment_bytes is not None:
            self.segment_bytes = int(segment_bytes)
        else:
            cfg = getattr(dataplane, "config", None)
            self.segment_bytes = int(
                getattr(cfg, "segment_bytes", _DEFAULT_SEGMENT_BYTES)
            )

        # Episode bookkeeping (set in reset()).
        self._last_path: int = -1
        self._last_goodput_mbps: float = 0.0
        self._last_rtt_ms: float = 0.0
        self._segments_done: int = 0

    @property
    def num_paths(self) -> int:
        return self.dataplane.num_paths

    @property
    def cap_mbps(self) -> float:
        return float(self.dataplane.cap_mbps)

    # -- Gym-like API ------------------------------------------------------- #

    def reset(self, *, seed: Optional[int] = None) -> Observation:
        self.dataplane.reset(seed=seed)
        self._last_path = -1
        self._last_goodput_mbps = 0.0
        self._last_rtt_ms = 0.0
        self._segments_done = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        action = int(action)
        if not 0 <= action < self.num_paths:
            raise ValueError(f"action {action} not in [0, {self.num_paths})")

        result = self.dataplane.download_segment(action, self.segment_bytes)
        reward = compute_reward(
            throughput_mbps=result.throughput_mbps,
            rtt_ms=result.rtt_ms,
            loss=result.loss,
            cap_mbps=self.cap_mbps,
            weights=self.reward_weights,
        )

        self._last_path = action
        self._last_goodput_mbps = result.throughput_mbps
        self._last_rtt_ms = result.rtt_ms
        self._segments_done += 1

        done = self.dataplane.is_done()
        info = {
            "chosen_path": action,
            "throughput_mbps": result.throughput_mbps,
            "rtt_ms": result.rtt_ms,
            "loss": result.loss,
            "duration_s": result.duration_s,
            "bytes_delivered": result.bytes_delivered,
            "clock_s": self.dataplane.clock_s,
        }
        return self.observe(), reward, done, info

    # -- Observation -------------------------------------------------------- #

    def observe(self) -> Observation:
        """Return ``{"global": (GLOBAL_DIM,), "paths": (N, PATH_FEATURE_DIM)}``."""
        stats = self.dataplane.current_path_stats()
        cap = max(self.cap_mbps, 1e-3)
        max_tp = max((s.throughput_mbps for s in stats), default=1e-3)
        max_tp = max(max_tp, 1e-3)

        paths = np.zeros((len(stats), PATH_FEATURE_DIM), dtype=np.float32)
        for i, s in enumerate(stats):
            paths[i, 0] = min(s.throughput_mbps / cap, 1.0)
            paths[i, 1] = min(RTT_TRUST_REF_MS, max(0.0, s.rtt_ms)) / RTT_TRUST_REF_MS
            paths[i, 2] = float(np.clip(s.loss, 0.0, 1.0))
            paths[i, 3] = s.throughput_mbps / max_tp
            paths[i, 4] = 1.0 if i == self._last_path else 0.0

        glob = np.zeros(GLOBAL_DIM, dtype=np.float32)
        glob[0] = self._progress()
        glob[1] = min(self._last_goodput_mbps / cap, 1.0)
        glob[2] = min(RTT_TRUST_REF_MS, max(0.0, self._last_rtt_ms)) / RTT_TRUST_REF_MS
        glob[3] = float(np.mean(paths[:, 0])) if len(stats) else 0.0

        return {"global": glob, "paths": paths}

    def current_path_stats(self) -> list[PathStats]:
        """Convenience passthrough used by baselines."""
        return self.dataplane.current_path_stats()

    # -- internals ---------------------------------------------------------- #

    def _progress(self) -> float:
        cfg = getattr(self.dataplane, "config", None)
        total = int(getattr(cfg, "episode_segments", 0))
        if total <= 0:
            return 0.0
        return min(1.0, self._segments_done / total)
