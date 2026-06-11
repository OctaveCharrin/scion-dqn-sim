"""
Adaptive-bitrate (ABR) building blocks for the Phase-2 video QoE environment.

These are the backend-agnostic pieces the env composes on top of any
:class:`~src.ns3env.dataplane.DataPlane`: a DASH-style **bitrate ladder**, a
**VMAF perceptual-quality** model, a client **playback buffer** (so rebuffering
becomes a delayed consequence of earlier choices -- the temporal structure that
makes RL worthwhile over a reactive heuristic), and a **QoE reward**.

Quality is measured with **VMAF** (0-100), not raw bitrate, because the
rate->quality relationship is concave/saturating: equal bitrate steps are *not*
equal perceptual steps. We map bitrate to VMAF with a logarithmic curve anchored
to the endpoints Netflix describes (low-quality encode ~20, high-quality ~95+;
~6 VMAF points ~= 1 JND; ~93+ is visually near-transparent). The QoE reward then
uses VMAF for both the quality term and the switching (smoothness) term:

    QoE = w_q * VMAF(R_k) - w_rebuf * rebuffer - w_switch * |VMAF(R_k) - VMAF(R_{k-1})|

normalized to ``[-1, 1]``. Replace :func:`vmaf_for` (or pass a measured per-rung
table) with content-specific VMAF when calibrating to a real encode.

Reference: Zhi Li, Christos Bampis, Julie Novak, Anne Aaron, Kyle Swanson, Anush
Moorthy, JD Cock, "VMAF: The Journey Continues," Netflix Technology Blog, 2018.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "BITRATE_LADDER_MBPS",
    "SEGMENT_DURATION_S",
    "BUFFER_MAX_S",
    "VMAF_MAX",
    "vmaf_for",
    "PlaybackBuffer",
    "QoEWeights",
    "compute_qoe_reward",
    "segment_bytes_for",
]

# Pensieve-style bitrate ladder (Mbps). The agent picks one rung per segment.
BITRATE_LADDER_MBPS: Tuple[float, ...] = (0.3, 0.75, 1.2, 1.85, 2.85, 4.3)

# DASH segment playback duration (s) and client buffer capacity (s).
SEGMENT_DURATION_S: float = 4.0
BUFFER_MAX_S: float = 60.0

# VMAF scale and the two (bitrate_mbps, VMAF) anchors defining the log rate-quality
# curve. Anchors follow the Netflix description (low encode ~20, high ~92); they
# are a documented stand-in for measured per-content VMAF, not exact values.
VMAF_MAX: float = 100.0
VMAF_ANCHOR_LOW: Tuple[float, float] = (0.3, 25.0)
VMAF_ANCHOR_HIGH: Tuple[float, float] = (4.3, 92.0)


def vmaf_for(
    bitrate_mbps: float,
    *,
    anchor_low: Tuple[float, float] = VMAF_ANCHOR_LOW,
    anchor_high: Tuple[float, float] = VMAF_ANCHOR_HIGH,
) -> float:
    """Map a bitrate (Mbps) to a VMAF score in ``[0, 100]`` (concave/log curve).

    ``VMAF(R) = a*ln(R) + b`` fit through the two anchors, clipped to ``[0, 100]``.
    Monotonically increasing and concave, capturing diminishing returns at high
    bitrate (saturation toward 100).
    """
    (r0, v0), (r1, v1) = anchor_low, anchor_high
    a = (v1 - v0) / (math.log(r1) - math.log(r0))
    b = v0 - a * math.log(r0)
    v = a * math.log(max(bitrate_mbps, 1e-6)) + b
    return float(min(VMAF_MAX, max(0.0, v)))


def segment_bytes_for(bitrate_mbps: float, segment_duration_s: float) -> int:
    """Encoded size of one ``segment_duration_s`` segment at ``bitrate_mbps``."""
    return int(round(bitrate_mbps * 1e6 * segment_duration_s / 8.0))


class PlaybackBuffer:
    """Client playback buffer (seconds of buffered video).

    Standard DASH model: while a segment downloads (taking ``download_time_s``)
    the buffer drains in real time; if it empties first the player **stalls**
    (rebuffers) for the remainder. On completion the segment adds
    ``segment_duration_s`` of video, capped at ``buffer_max_s``.
    """

    def __init__(
        self,
        *,
        segment_duration_s: float = SEGMENT_DURATION_S,
        buffer_max_s: float = BUFFER_MAX_S,
        initial_s: float = 0.0,
    ) -> None:
        self.segment_duration_s = float(segment_duration_s)
        self.buffer_max_s = float(buffer_max_s)
        self.initial_s = float(initial_s)
        self.buffer_s = self.initial_s

    def reset(self) -> None:
        self.buffer_s = self.initial_s

    def update(self, download_time_s: float) -> float:
        """Apply one segment download; return the rebuffering time (s) incurred."""
        dl = max(0.0, float(download_time_s))
        rebuffer_s = max(0.0, dl - self.buffer_s)
        self.buffer_s = max(0.0, self.buffer_s - dl) + self.segment_duration_s
        self.buffer_s = min(self.buffer_s, self.buffer_max_s)
        return rebuffer_s


@dataclass
class QoEWeights:
    """Weights for the VMAF-based linear QoE reward.

    ``w_quality`` scales perceptual quality (VMAF), ``w_rebuffer`` penalizes stalls
    (per segment-duration stalled), ``w_switch`` penalizes VMAF changes between
    consecutive segments. Defaults balance them so a full-segment stall roughly
    cancels a top-quality segment.
    """

    w_quality: float = 1.0
    w_rebuffer: float = 1.0
    w_switch: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]] = None) -> "QoEWeights":
        base = cls()
        if not data:
            return base
        for key in ("w_quality", "w_rebuffer", "w_switch"):
            if key in data:
                setattr(base, key, float(data[key]))
        return base


def compute_qoe_reward(
    *,
    bitrate_mbps: float,
    prev_bitrate_mbps: float,
    rebuffer_s: float,
    segment_duration_s: float = SEGMENT_DURATION_S,
    weights: Optional[QoEWeights] = None,
    quality_fn: Callable[[float], float] = vmaf_for,
) -> float:
    """Normalized VMAF-based QoE reward in ``[-1, 1]``.

    ``r = w_quality*q - w_rebuffer*rebuf - w_switch*switch`` where the quality and
    switching terms use VMAF: ``q = VMAF(R)/100``, ``switch = |VMAF(R) -
    VMAF(R_prev)|/100``; ``rebuf = rebuffer_s / segment_duration`` (a full-segment
    stall costs ``w_rebuffer``).
    """
    w = weights or QoEWeights()
    seg = max(segment_duration_s, 1e-6)

    q = quality_fn(bitrate_mbps) / VMAF_MAX
    prev_q = quality_fn(prev_bitrate_mbps) / VMAF_MAX
    rebuf = max(0.0, rebuffer_s) / seg
    switch = abs(q - prev_q)

    reward = w.w_quality * q - w.w_rebuffer * rebuf - w.w_switch * switch
    return max(-1.0, min(1.0, reward))


def qoe_components(
    *,
    bitrate_mbps: float,
    prev_bitrate_mbps: float,
    rebuffer_s: float,
    quality_fn: Callable[[float], float] = vmaf_for,
) -> Dict[str, float]:
    """Unweighted VMAF QoE terms for logging/inspection."""
    vmaf = quality_fn(bitrate_mbps)
    return {
        "vmaf": vmaf,
        "rebuffer_s": max(0.0, float(rebuffer_s)),
        "switch_vmaf": abs(vmaf - quality_fn(prev_bitrate_mbps)),
    }


def ladder_vmaf(
    ladder: Sequence[float] = BITRATE_LADDER_MBPS,
    quality_fn: Callable[[float], float] = vmaf_for,
) -> Tuple[float, ...]:
    """VMAF score for each rung of ``ladder`` (precompute for the env)."""
    return tuple(quality_fn(b) for b in ladder)
