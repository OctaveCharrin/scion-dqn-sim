"""
Heuristic ABR bitrate selectors for the Phase-2 video QoE environment.

Interface mirrors the multipath selectors: ``select(obs) -> int`` returning a
bitrate-candidate index, reading the observation produced by
:class:`~src.ns3env.abr_env.VideoAbrEnv` (candidate rows are bitrate options;
column meanings in ``abr_env``). These are the comparison points the learned
agent must beat: rate-based and buffer-based (BOLA/BBA-style) are the classic ABR
heuristics, with always-max / always-min as trivial bounds.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.ns3env.abr_env import C_FEASIBLE, G_BUFFER

Observation = Dict[str, np.ndarray]


class FixedBitrateSelector:
    """Always pick the rung at ``frac`` of the ladder (0 = lowest, 1 = highest)."""

    def __init__(self, frac: float = 0.5) -> None:
        self.frac = float(np.clip(frac, 0.0, 1.0))

    def reset(self) -> None:
        pass

    def select(self, obs: Observation) -> int:
        n = obs["paths"].shape[0]
        if n == 0:
            return 0
        return int(round(self.frac * (n - 1)))


class RateBasedSelector:
    """Pick the highest bitrate the current throughput estimate can sustain.

    Uses the per-candidate feasibility column (``1.0`` exactly when the estimated
    throughput meets/exceeds that bitrate); falls back to the lowest rung if none
    are sustainable.
    """

    def reset(self) -> None:
        pass

    def select(self, obs: Observation) -> int:
        paths = obs["paths"]
        if paths.shape[0] == 0:
            return 0
        sustainable = np.flatnonzero(paths[:, C_FEASIBLE] >= 1.0)
        return int(sustainable[-1]) if sustainable.size else 0


class BufferBasedSelector:
    """BBA/BOLA-style: map the current buffer level to a bitrate rung.

    Below ``reservoir`` (fraction of max buffer) pick the lowest rung; above
    ``reservoir + cushion`` pick the highest; linearly interpolate in between.
    """

    def __init__(self, reservoir: float = 0.15, cushion: float = 0.6) -> None:
        self.reservoir = float(reservoir)
        self.cushion = max(float(cushion), 1e-6)

    def reset(self) -> None:
        pass

    def select(self, obs: Observation) -> int:
        n = obs["paths"].shape[0]
        if n == 0:
            return 0
        buf = float(obs["global"][G_BUFFER])
        frac = (buf - self.reservoir) / self.cushion
        frac = float(np.clip(frac, 0.0, 1.0))
        return int(round(frac * (n - 1)))


#: Registry mirroring ``MULTIPATH_SELECTORS``.
ABR_SELECTORS = {
    "rate_based": RateBasedSelector,
    "buffer_based": BufferBasedSelector,
    "always_max": lambda: FixedBitrateSelector(1.0),
    "always_min": lambda: FixedBitrateSelector(0.0),
    "fixed_mid": lambda: FixedBitrateSelector(0.5),
}


# --------------------------------------------------------------------------- #
# Joint (path x bitrate) selectors for the multipath ABR env.
# --------------------------------------------------------------------------- #

from src.ns3env.abr_joint_env import (  # noqa: E402  (avoid import cycle at top)
    CJ_FEASIBLE,
    CJ_PATH_RTT,
    CJ_PATH_TP,
    GJ_BUFFER,
)


class JointPathBitrateSelector:
    """Composable heuristic: pick a path, then a bitrate on it.

    ``path_mode``: ``"throughput"`` (highest-capacity path) or ``"minrtt"``.
    ``bitrate_mode``: ``"rate"`` (highest sustainable rung on the chosen path),
    ``"buffer"`` (BBA/BOLA-style buffer->rung map), or ``"fixed"`` (``frac`` rung).
    Returns the path-major flattened index ``path * num_bitrates + bitrate``.
    """

    def __init__(
        self,
        num_bitrates: int,
        *,
        path_mode: str = "throughput",
        bitrate_mode: str = "rate",
        frac: float = 1.0,
        reservoir: float = 0.15,
        cushion: float = 0.6,
    ) -> None:
        self.b = int(num_bitrates)
        self.path_mode = path_mode
        self.bitrate_mode = bitrate_mode
        self.frac = float(np.clip(frac, 0.0, 1.0))
        self.reservoir = float(reservoir)
        self.cushion = max(float(cushion), 1e-6)

    def reset(self) -> None:
        pass

    def select(self, obs: Observation) -> int:
        paths = obs["paths"]
        total = paths.shape[0]
        if total == 0:
            return 0
        n = max(1, total // self.b)
        cube = paths[: n * self.b].reshape(n, self.b, -1)

        if self.path_mode == "minrtt":
            p = int(np.argmin(cube[:, 0, CJ_PATH_RTT]))
        else:
            p = int(np.argmax(cube[:, 0, CJ_PATH_TP]))

        if self.bitrate_mode == "rate":
            sustainable = np.flatnonzero(cube[p, :, CJ_FEASIBLE] >= 1.0)
            b = int(sustainable[-1]) if sustainable.size else 0
        elif self.bitrate_mode == "buffer":
            buf = float(obs["global"][GJ_BUFFER])
            frac = float(np.clip((buf - self.reservoir) / self.cushion, 0.0, 1.0))
            b = int(round(frac * (self.b - 1)))
        else:  # fixed
            b = int(round(self.frac * (self.b - 1)))

        return p * self.b + b


#: Joint registry; values are factories taking ``num_bitrates``.
ABR_JOINT_SELECTORS = {
    "bestpath_rate": lambda b: JointPathBitrateSelector(
        b, path_mode="throughput", bitrate_mode="rate"
    ),
    "bestpath_buffer": lambda b: JointPathBitrateSelector(
        b, path_mode="throughput", bitrate_mode="buffer"
    ),
    "minrtt_rate": lambda b: JointPathBitrateSelector(
        b, path_mode="minrtt", bitrate_mode="rate"
    ),
    "bestpath_max": lambda b: JointPathBitrateSelector(
        b, path_mode="throughput", bitrate_mode="fixed", frac=1.0
    ),
    "bestpath_min": lambda b: JointPathBitrateSelector(
        b, path_mode="throughput", bitrate_mode="fixed", frac=0.0
    ),
}
