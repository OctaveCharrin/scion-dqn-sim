"""
Phase-2 adaptive-bitrate (ABR) video environment over a pluggable data plane.

The agent chooses a **bitrate** per segment; the env tracks a client playback
buffer and returns a **VMAF-based QoE reward** (perceptual quality - rebuffer -
quality switching). This is the single-path setting (canonical Pensieve ABR): the
segment is fetched over a fixed path, so the only decision is quality. Phase-2
step 2 extends the candidate set to (path, bitrate) pairs for joint multipath ABR.

Observation keeps the scoring contract ``{"global": (G,), "paths": (B, P)}`` so
``EnhancedPathScoringDQNAgent`` is reusable unchanged -- here each "path" row is a
**bitrate candidate** and the action selects one. Candidate/global feature columns
are exposed as constants so the heuristic baselines read the same observation.

Quality features and the reward use VMAF (see :mod:`src.ns3env.abr`); throughput /
feasibility / buffer features use *bitrate* (download time depends on encoded
bytes, not on the perceptual score).
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

Observation = Dict[str, np.ndarray]

# Global feature columns.
G_BUFFER = 0  # buffer level / buffer_max
G_LAST_QUALITY = 1  # last chosen VMAF / 100
G_PROGRESS = 2  # segments done / episode_segments
G_THROUGHPUT = 3  # throughput EWMA / cap
GLOBAL_DIM = 4

# Per-candidate (bitrate option) feature columns.
C_QUALITY = 0  # VMAF(bitrate) / 100  (perceptual quality, concave in bitrate)
C_FEASIBLE = 1  # min(1, throughput_est / bitrate): 1 == path comfortably supports it
C_BUFFER_DELTA = 2  # predicted (segment_dur - est_download_time)/buffer_max, clipped
C_SWITCH = 3  # |VMAF(bitrate) - last_VMAF| / 100
C_IS_LAST = 4  # 1 if this is the previously chosen bitrate
PATH_FEATURE_DIM = 5


class VideoAbrEnv:
    """Single-path adaptive-bitrate env: action = bitrate index, reward = VMAF QoE."""

    def __init__(
        self,
        dataplane: DataPlane,
        *,
        ladder_mbps: Sequence[float] = BITRATE_LADDER_MBPS,
        segment_duration_s: float = SEGMENT_DURATION_S,
        buffer_max_s: float = BUFFER_MAX_S,
        qoe_weights: Optional[QoEWeights] = None,
        path_idx: int = 0,
        throughput_ewma: float = 0.5,
    ):
        self.dataplane = dataplane
        self.ladder = tuple(float(b) for b in ladder_mbps)
        self.ladder_vmaf = ladder_vmaf(self.ladder)
        self.segment_duration_s = float(segment_duration_s)
        self.qoe_weights = qoe_weights or QoEWeights()
        self.path_idx = int(path_idx)
        self._tp_alpha = float(throughput_ewma)
        self.buffer = PlaybackBuffer(
            segment_duration_s=segment_duration_s, buffer_max_s=buffer_max_s
        )

        # Scoring-agent dimensions (read by the trainer).
        self.global_dim = GLOBAL_DIM
        self.path_dim = PATH_FEATURE_DIM
        self.num_actions = len(self.ladder)

        self._last_bitrate_idx = 0
        self._throughput_est = 0.0
        self._segments_done = 0

    @property
    def cap_mbps(self) -> float:
        return float(self.dataplane.cap_mbps)

    # -- Gym-like API ------------------------------------------------------- #

    def reset(self, *, seed: Optional[int] = None) -> Observation:
        self.dataplane.reset(seed=seed)
        self.buffer.reset()
        # Start at the lowest rung (a safe DASH startup) and seed the throughput
        # estimate from the fixed path's current stat.
        self._last_bitrate_idx = 0
        stats = self.dataplane.current_path_stats()
        self._throughput_est = (
            float(stats[self.path_idx].throughput_mbps)
            if 0 <= self.path_idx < len(stats)
            else self.cap_mbps
        )
        self._segments_done = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        idx = int(action)
        if not 0 <= idx < self.num_actions:
            raise ValueError(f"action {idx} not in [0, {self.num_actions})")

        bitrate = self.ladder[idx]
        prev_bitrate = self.ladder[self._last_bitrate_idx]
        seg_bytes = segment_bytes_for(bitrate, self.segment_duration_s)

        result = self.dataplane.download_segment(self.path_idx, seg_bytes)
        rebuffer_s = self.buffer.update(result.duration_s)

        reward = compute_qoe_reward(
            bitrate_mbps=bitrate,
            prev_bitrate_mbps=prev_bitrate,
            rebuffer_s=rebuffer_s,
            segment_duration_s=self.segment_duration_s,
            weights=self.qoe_weights,
        )

        # EWMA the realized throughput for the next decision's estimate.
        self._throughput_est = (
            1.0 - self._tp_alpha
        ) * self._throughput_est + self._tp_alpha * result.throughput_mbps
        self._last_bitrate_idx = idx
        self._segments_done += 1

        done = self.dataplane.is_done()
        info = {
            "chosen_bitrate_mbps": bitrate,
            "chosen_vmaf": self.ladder_vmaf[idx],
            "bitrate_idx": idx,
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
        cap = max(self.cap_mbps, 1e-3)
        est = max(self._throughput_est, 1e-6)
        last_vmaf = self.ladder_vmaf[self._last_bitrate_idx]

        cands = np.zeros((self.num_actions, PATH_FEATURE_DIM), dtype=np.float32)
        for i, bitrate in enumerate(self.ladder):
            est_download_s = (bitrate * self.segment_duration_s) / est
            buf_delta = (
                self.segment_duration_s - est_download_s
            ) / self.buffer.buffer_max_s
            cands[i, C_QUALITY] = self.ladder_vmaf[i] / VMAF_MAX
            cands[i, C_FEASIBLE] = min(1.0, est / bitrate)
            cands[i, C_BUFFER_DELTA] = float(np.clip(buf_delta, -1.0, 1.0))
            cands[i, C_SWITCH] = abs(self.ladder_vmaf[i] - last_vmaf) / VMAF_MAX
            cands[i, C_IS_LAST] = 1.0 if i == self._last_bitrate_idx else 0.0

        glob = np.zeros(GLOBAL_DIM, dtype=np.float32)
        glob[G_BUFFER] = min(self.buffer.buffer_s / self.buffer.buffer_max_s, 1.0)
        glob[G_LAST_QUALITY] = last_vmaf / VMAF_MAX
        glob[G_PROGRESS] = self._progress()
        glob[G_THROUGHPUT] = min(est / cap, 1.0)

        return {"global": glob, "paths": cands}

    # -- internals ---------------------------------------------------------- #

    def _progress(self) -> float:
        cfg = getattr(self.dataplane, "config", None)
        total = int(getattr(cfg, "episode_segments", 0))
        if total <= 0:
            return 0.0
        return min(1.0, self._segments_done / total)
