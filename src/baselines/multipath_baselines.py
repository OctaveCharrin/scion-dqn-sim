"""
Heuristic path selectors for the multipath-QUIC video environment.

These are the ``ns3`` branch analogue of the SCION baselines in this package.
The SCION selectors expose ``select_path(paths, metrics, flow, state) -> int``;
here the natural input is the environment observation dict
(``{"global": (G,), "paths": (N, P)}`` from :mod:`src.ns3env.video_env`), so the
interface is ``select(obs) -> int``. Register new ones in the evaluation script
just like the SCION baselines are registered in ``05_evaluate_methods.py``.

Path-feature column order (see ``video_env.PATH_FEATURE_DIM``):
    0: throughput_norm   1: rtt_norm   2: loss   3: throughput_ratio   4: last_chosen
"""

from __future__ import annotations

from typing import Dict

import numpy as np

Observation = Dict[str, np.ndarray]

_TP = 0
_RTT = 1


class RoundRobinSelector:
    """Cycle through paths regardless of state (multipath default scheduler)."""

    def __init__(self) -> None:
        self._next = 0

    def reset(self) -> None:
        self._next = 0

    def select(self, obs: Observation) -> int:
        n = obs["paths"].shape[0]
        if n == 0:
            return 0
        choice = self._next % n
        self._next = (self._next + 1) % n
        return int(choice)


class MinRTTSelector:
    """Pick the lowest-RTT path -- the classic MPTCP/MPQUIC scheduler heuristic."""

    def select(self, obs: Observation) -> int:
        paths = obs["paths"]
        if paths.shape[0] == 0:
            return 0
        return int(np.argmin(paths[:, _RTT]))


class MaxThroughputSelector:
    """Greedy: pick the highest currently-observed throughput path."""

    def select(self, obs: Observation) -> int:
        paths = obs["paths"]
        if paths.shape[0] == 0:
            return 0
        return int(np.argmax(paths[:, _TP]))


class StaticPathSelector:
    """Always use a fixed path index (single-path baseline)."""

    def __init__(self, path_idx: int = 0) -> None:
        self.path_idx = int(path_idx)

    def reset(self) -> None:  # symmetry with stateful selectors
        pass

    def select(self, obs: Observation) -> int:
        n = obs["paths"].shape[0]
        if n == 0:
            return 0
        return min(self.path_idx, n - 1)


class SingleBestPathSelector:
    """Lock onto the best path observed on the first step, then stick to it.

    A no-adaptation single-path policy that still makes a sensible initial
    choice -- a stronger single-path baseline than a fixed index.
    """

    def __init__(self) -> None:
        self._locked: int | None = None

    def reset(self) -> None:
        self._locked = None

    def select(self, obs: Observation) -> int:
        paths = obs["paths"]
        if paths.shape[0] == 0:
            return 0
        if self._locked is None:
            self._locked = int(np.argmax(paths[:, _TP]))
        return min(self._locked, paths.shape[0] - 1)


#: Convenience registry mirroring how SCION baselines are looked up by name.
MULTIPATH_SELECTORS = {
    "round_robin": RoundRobinSelector,
    "min_rtt": MinRTTSelector,
    "max_throughput": MaxThroughputSelector,
    "static_path0": StaticPathSelector,
    "single_best": SingleBestPathSelector,
}
