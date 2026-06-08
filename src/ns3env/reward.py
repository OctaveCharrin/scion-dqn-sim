"""
Reward for the multipath-QUIC video environment.

Phase 1 uses a goodput + trust composite that mirrors the SCION
``EvaluationPathSelectionEnv.compute_reward`` so the existing reward machinery
(``RewardWeights``, ``src.rl.reward_profiles``, the weight-FiLM conditional agent)
applies unchanged. Phase 2 will extend this with QoE terms (rebuffering, bitrate,
bitrate-switching) while keeping the same weighted-objective shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

__all__ = ["RewardWeights", "compute_reward", "RTT_TRUST_REF_MS"]

# Reference RTT (ms) at which the delay component of trust saturates.
RTT_TRUST_REF_MS = 200.0


@dataclass
class RewardWeights:
    """Composite goodput + trust reward weights.

    Field-compatible with the SCION ``EvaluationPathSelectionEnv.RewardWeights``
    so ``src.rl.reward_profiles`` and the weight-FiLM conditional agent remain
    usable on this branch -- but defined here to keep ``src.ns3env`` free of the
    heavy SCION import chain (networkx, topology, ...).
    """

    w1: float = 0.7  # goodput
    w2: float = 0.3  # trust
    w3: float = 0.5  # trust sensitivity to loss
    w4: float = 0.5  # trust sensitivity to delay
    w_probe: float = 0.05  # probe-cost weight (reserved for later phases)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]] = None) -> "RewardWeights":
        base = cls()
        if not data:
            return base
        for key in ("w1", "w2", "w3", "w4", "w_probe"):
            if key in data:
                setattr(base, key, float(data[key]))
        return base


def compute_reward(
    *,
    throughput_mbps: float,
    rtt_ms: float,
    loss: float,
    cap_mbps: float,
    weights: Optional[RewardWeights] = None,
) -> float:
    """Composite reward in ``[-1, 1]``.

    ``r = 2 * (w1 * goodput + w2 * trust) - 1``

    * ``goodput = min(throughput / cap, 1)`` -- normalized realized throughput.
    * ``trust   = 1 - (w3 * loss + w4 * rtt_norm)`` -- delay/loss quality, clipped.

    Args:
        throughput_mbps: realized goodput of the chosen path this segment.
        rtt_ms: smoothed RTT of the chosen path.
        loss: loss rate in ``[0, 1]``.
        cap_mbps: normalization cap (``DataPlane.cap_mbps``).
        weights: reward weights; defaults to :class:`RewardWeights` defaults.
    """
    w = weights or RewardWeights()

    goodput = 1.0 if cap_mbps <= 1e-3 else min(throughput_mbps / cap_mbps, 1.0)
    rtt_norm = min(RTT_TRUST_REF_MS, max(0.0, rtt_ms)) / RTT_TRUST_REF_MS
    trust = max(0.0, min(1.0, 1.0 - (w.w3 * loss + w.w4 * rtt_norm)))

    reward = 2.0 * (w.w1 * goodput + w.w2 * trust) - 1.0
    return max(-1.0, min(1.0, reward))
