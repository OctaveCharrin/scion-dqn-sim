"""
Phase-2 step 2: joint (path x bitrate) multipath ABR environment.

Extends the single-path ABR env to the project's real target: each decision
chooses **both** which path to fetch the segment on **and** at what bitrate. Every
``(path, bitrate)`` pair is presented as one scoring candidate, so the *same*
``EnhancedPathScoringDQNAgent`` jointly optimizes path selection and quality --
no agent change, just a larger candidate set (``num_paths * num_bitrates`` rows).

Reward and buffer dynamics are identical to :mod:`src.ns3env.abr_env` (VMAF-based
QoE over a playback buffer). Per-path features come straight from the data plane's
``current_path_stats`` (expected capacity on the mock; EWMA on NS-3), so the env
works against both backends unchanged.

Candidate rows are ordered **path-major**: row ``p*num_bitrates + b`` is
``(path p, bitrate b)``; the chosen action index decodes the same way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.ns3env.abr import (
    BITRATE_LADDER_MBPS,
    BUFFER_MAX_S,
    SEGMENT_DURATION_S,
    VMAF_MAX,
    PlaybackBuffer,
    QoEWeights,
    compute_qoe_reward,
    ladder_vmaf,
    segment_bytes_for,
)
from src.ns3env.dataplane import DataPlane
from src.ns3env.reward import RTT_TRUST_REF_MS

Observation = Dict[str, np.ndarray]

# Global feature columns.
GJ_BUFFER = 0
GJ_LAST_QUALITY = 1
GJ_PROGRESS = 2
GJ_MEAN_THROUGHPUT = 3
GLOBAL_DIM = 4

# Per-candidate (path, bitrate) feature columns.
CJ_QUALITY = 0  # VMAF(bitrate)/100
CJ_FEASIBLE = 1  # min(1, path_throughput / bitrate)
CJ_BUFFER_DELTA = 2  # predicted (seg_dur - est_download_on_path)/buffer_max, clipped
CJ_SWITCH = 3  # |VMAF(bitrate) - last_VMAF|/100
CJ_PATH_TP = 4  # path throughput / cap
CJ_PATH_RTT = 5  # path rtt_norm
CJ_PATH_LOSS = 6  # path loss
CJ_IS_LAST_PATH = 7  # 1 if this path was last chosen
CJ_IS_LAST_BR = 8  # 1 if this bitrate was last chosen
PATH_FEATURE_DIM = 9


class VideoMultipathAbrEnv:
    """Joint multipath ABR: action selects a ``(path, bitrate)`` candidate."""

    def __init__(
        self,
        dataplane: DataPlane,
        *,
        ladder_mbps: Sequence[float] = BITRATE_LADDER_MBPS,
        segment_duration_s: float = SEGMENT_DURATION_S,
        buffer_max_s: float = BUFFER_MAX_S,
        qoe_weights: Optional[QoEWeights] = None,
    ):
        self.dataplane = dataplane
        self.ladder = tuple(float(b) for b in ladder_mbps)
        self.ladder_vmaf = ladder_vmaf(self.ladder)
        self.segment_duration_s = float(segment_duration_s)
        self.qoe_weights = qoe_weights or QoEWeights()
        self.buffer = PlaybackBuffer(
            segment_duration_s=segment_duration_s, buffer_max_s=buffer_max_s
        )

        self.num_paths = int(dataplane.num_paths)
        self.num_bitrates = len(self.ladder)
        self.num_actions = self.num_paths * self.num_bitrates
        self.global_dim = GLOBAL_DIM
        self.path_dim = PATH_FEATURE_DIM

        self._last_path = -1
        self._last_bitrate_idx = 0
        self._segments_done = 0

    @property
    def cap_mbps(self) -> float:
        return float(self.dataplane.cap_mbps)

    def _decode(self, action: int) -> Tuple[int, int]:
        return action // self.num_bitrates, action % self.num_bitrates

    # -- Gym-like API ------------------------------------------------------- #

    def reset(self, *, seed: Optional[int] = None) -> Observation:
        self.dataplane.reset(seed=seed)
        self.buffer.reset()
        self._last_path = -1
        self._last_bitrate_idx = 0
        self._segments_done = 0
        # num_paths may be confirmed by the data plane after reset (NS-3).
        self.num_paths = int(self.dataplane.num_paths)
        self.num_actions = self.num_paths * self.num_bitrates
        return self.observe()

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        a = int(action)
        if not 0 <= a < self.num_actions:
            raise ValueError(f"action {a} not in [0, {self.num_actions})")
        path_idx, bitrate_idx = self._decode(a)

        bitrate = self.ladder[bitrate_idx]
        prev_bitrate = self.ladder[self._last_bitrate_idx]
        seg_bytes = segment_bytes_for(bitrate, self.segment_duration_s)

        result = self.dataplane.download_segment(path_idx, seg_bytes)
        rebuffer_s = self.buffer.update(result.duration_s)

        reward = compute_qoe_reward(
            bitrate_mbps=bitrate,
            prev_bitrate_mbps=prev_bitrate,
            rebuffer_s=rebuffer_s,
            segment_duration_s=self.segment_duration_s,
            weights=self.qoe_weights,
        )

        self._last_path = path_idx
        self._last_bitrate_idx = bitrate_idx
        self._segments_done += 1

        done = self.dataplane.is_done()
        info = {
            "chosen_path": path_idx,
            "chosen_bitrate_mbps": bitrate,
            "chosen_vmaf": self.ladder_vmaf[bitrate_idx],
            "bitrate_idx": bitrate_idx,
            "rebuffer_s": rebuffer_s,
            "buffer_s": self.buffer.buffer_s,
            "throughput_mbps": result.throughput_mbps,
            "download_s": result.duration_s,
            "segment_bytes": seg_bytes,
            "clock_s": self.dataplane.clock_s,
        }
        return self.observe(), reward, done, info

    # -- Observation -------------------------------------------------------- #

    def observe(self) -> Observation:
        stats = self.dataplane.current_path_stats()
        cap = max(self.cap_mbps, 1e-3)
        last_vmaf = self.ladder_vmaf[self._last_bitrate_idx]
        buf_max = self.buffer.buffer_max_s

        n = len(stats)
        cands = np.zeros((n * self.num_bitrates, PATH_FEATURE_DIM), dtype=np.float32)
        for p in range(n):
            tp = max(stats[p].throughput_mbps, 1e-6)
            rtt_norm = (
                min(RTT_TRUST_REF_MS, max(0.0, stats[p].rtt_ms)) / RTT_TRUST_REF_MS
            )
            loss = float(np.clip(stats[p].loss, 0.0, 1.0))
            for b, bitrate in enumerate(self.ladder):
                row = p * self.num_bitrates + b
                est_download_s = (bitrate * self.segment_duration_s) / tp
                buf_delta = (self.segment_duration_s - est_download_s) / buf_max
                cands[row, CJ_QUALITY] = self.ladder_vmaf[b] / VMAF_MAX
                cands[row, CJ_FEASIBLE] = min(1.0, tp / bitrate)
                cands[row, CJ_BUFFER_DELTA] = float(np.clip(buf_delta, -1.0, 1.0))
                cands[row, CJ_SWITCH] = abs(self.ladder_vmaf[b] - last_vmaf) / VMAF_MAX
                cands[row, CJ_PATH_TP] = min(tp / cap, 1.0)
                cands[row, CJ_PATH_RTT] = rtt_norm
                cands[row, CJ_PATH_LOSS] = loss
                cands[row, CJ_IS_LAST_PATH] = 1.0 if p == self._last_path else 0.0
                cands[row, CJ_IS_LAST_BR] = 1.0 if b == self._last_bitrate_idx else 0.0

        glob = np.zeros(GLOBAL_DIM, dtype=np.float32)
        glob[GJ_BUFFER] = min(self.buffer.buffer_s / buf_max, 1.0)
        glob[GJ_LAST_QUALITY] = last_vmaf / VMAF_MAX
        glob[GJ_PROGRESS] = self._progress()
        mean_tp = float(np.mean([s.throughput_mbps for s in stats])) if stats else 0.0
        glob[GJ_MEAN_THROUGHPUT] = min(mean_tp / cap, 1.0)

        return {"global": glob, "paths": cands}

    # -- internals ---------------------------------------------------------- #

    def _progress(self) -> float:
        cfg = getattr(self.dataplane, "config", None)
        total = int(getattr(cfg, "episode_segments", 0))
        if total <= 0:
            return 0.0
        return min(1.0, self._segments_done / total)
