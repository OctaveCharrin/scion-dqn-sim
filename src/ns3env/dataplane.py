"""
Data-plane abstraction for the multipath-QUIC video environment.

A :class:`DataPlane` represents *the network* as seen by the decision/abstraction
layer. The decision epoch is **one video segment download**: the agent picks a
path, the data plane simulates delivering one segment over it, advances its
internal clock, and reports what happened plus the (now-current) per-path stats.

Two backends implement the same interface:

* :class:`MockTraceDataPlane` -- pure Python, trace-driven. Each path has a
  time-varying capacity (sinusoid + cross-traffic + noise). Deterministic given a
  seed. Runs on Windows/CI with no NS-3 dependency, so the env, reward, agents and
  baselines are fully unit-testable without compiling NS-3.
* :class:`Ns3DataPlane` -- a thin stub here; drives a real NS-3 scenario over the
  ns3-ai shared-memory bridge (Linux/WSL2). The method contract is identical so
  the env code is backend-agnostic.

Design note: keeping *all* network modelling behind this interface is what lets
RL development proceed independently of the C++ / ns3-ai integration risk.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class PathStats:
    """Per-path network state observable by the agent at a decision point.

    These are the quantities a real client could measure/estimate per subflow
    (an EWMA throughput estimate, smoothed RTT, loss estimate).
    """

    throughput_mbps: float
    rtt_ms: float
    loss: float


@dataclass
class DownloadResult:
    """Outcome of delivering one segment over the chosen path."""

    path_idx: int
    throughput_mbps: float  # realized goodput for this segment
    rtt_ms: float
    loss: float
    duration_s: float  # sim time the download took
    bytes_delivered: int


class DataPlane(ABC):
    """Backend-agnostic network model. Decision epoch = one segment download."""

    #: Number of candidate paths (subflows). The path-scoring agent handles
    #: variable counts, but for a given episode this is fixed.
    num_paths: int

    #: Normalization cap (Mbps): an upper bound on achievable per-path throughput,
    #: used to normalize observations and the goodput reward (mirrors the SCION
    #: ``goodput_cap`` idea).
    cap_mbps: float

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None) -> None:
        """Start a new episode (rewind clock, optionally reseed)."""

    @abstractmethod
    def current_path_stats(self) -> List[PathStats]:
        """Observable per-path stats *now* (length == ``num_paths``)."""

    @abstractmethod
    def download_segment(self, path_idx: int, segment_bytes: int) -> DownloadResult:
        """Deliver one segment over ``path_idx`` and advance the clock."""

    @abstractmethod
    def is_done(self) -> bool:
        """True once the episode horizon is exhausted."""

    @property
    @abstractmethod
    def clock_s(self) -> float:
        """Current sim time in seconds (for observation/time features)."""


# --------------------------------------------------------------------------- #
# Mock trace-driven backend
# --------------------------------------------------------------------------- #


@dataclass
class _PathTrace:
    """Parametric time-varying capacity model for one path."""

    base_mbps: float
    amp: float  # relative sinusoid amplitude in [0, 1)
    period_s: float
    phase: float
    base_rtt_ms: float
    noise_std: float

    def capacity_mbps(self, t: float, rng: np.random.Generator) -> float:
        season = 1.0 + self.amp * math.sin(
            2.0 * math.pi * t / self.period_s + self.phase
        )
        noise = 1.0 + rng.normal(0.0, self.noise_std)
        return max(0.05, self.base_mbps * season * noise)

    def rtt_ms(self, capacity_mbps: float) -> float:
        # Queueing grows as the path nears its own (de-seasoned) baseline.
        util = min(1.5, self.base_mbps / max(capacity_mbps, 0.05))
        return self.base_rtt_ms * (1.0 + 0.25 * max(0.0, util - 1.0))

    @staticmethod
    def loss(capacity_mbps: float, base_mbps: float) -> float:
        # Light loss when a path is congested below ~30% of its baseline.
        ratio = capacity_mbps / max(base_mbps, 0.05)
        if ratio >= 0.5:
            return 0.0
        return float(min(0.1, 0.1 * (0.5 - ratio) / 0.5))


@dataclass
class MockTraceConfig:
    """Configuration for :class:`MockTraceDataPlane`.

    Defaults describe a deliberately *asymmetric, time-varying* 3-path scenario
    (e.g. wired / Wi-Fi / LTE) so that a fixed single-path or round-robin policy
    is suboptimal and a state-aware agent can win.
    """

    num_paths: int = 3
    episode_segments: int = 48
    segment_bytes: int = 500_000  # ~4 s of ~1 Mbps video
    base_mbps: Sequence[float] = (6.0, 3.0, 1.5)
    amp: Sequence[float] = (0.5, 0.7, 0.4)
    period_s: Sequence[float] = (40.0, 17.0, 90.0)
    base_rtt_ms: Sequence[float] = (20.0, 35.0, 60.0)
    noise_std: float = 0.05
    seed: int = 0

    def _per_path(self, values: Sequence[float]) -> List[float]:
        """Broadcast a scalar/short sequence to one value per path."""
        vals = list(values)
        if len(vals) == 1:
            return vals * self.num_paths
        if len(vals) < self.num_paths:
            # Repeat cyclically to fill.
            return [vals[i % len(vals)] for i in range(self.num_paths)]
        return vals[: self.num_paths]


class MockTraceDataPlane(DataPlane):
    """Trace-driven, deterministic-per-seed network for fast RL iteration/tests."""

    def __init__(self, config: Optional[MockTraceConfig] = None):
        self.config = config or MockTraceConfig()
        self.num_paths = int(self.config.num_paths)
        base = self.config._per_path(self.config.base_mbps)
        amp = self.config._per_path(self.config.amp)
        period = self.config._per_path(self.config.period_s)
        rtt = self.config._per_path(self.config.base_rtt_ms)
        # Spread phases so paths peak at different times.
        self._traces: List[_PathTrace] = [
            _PathTrace(
                base_mbps=base[i],
                amp=amp[i],
                period_s=period[i],
                phase=2.0 * math.pi * i / max(1, self.num_paths),
                base_rtt_ms=rtt[i],
                noise_std=self.config.noise_std,
            )
            for i in range(self.num_paths)
        ]
        # Cap = peak achievable across paths (used for goodput normalization).
        self.cap_mbps = max(t.base_mbps * (1.0 + t.amp) for t in self._traces)
        self._rng = np.random.default_rng(self.config.seed)
        self._t = 0.0
        self._segments_done = 0

    # -- DataPlane API ------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._segments_done = 0

    @property
    def clock_s(self) -> float:
        return self._t

    def current_path_stats(self) -> List[PathStats]:
        return [self._stats_at(i, self._t) for i in range(self.num_paths)]

    def download_segment(self, path_idx: int, segment_bytes: int) -> DownloadResult:
        if not 0 <= path_idx < self.num_paths:
            raise IndexError(f"path_idx {path_idx} out of range [0, {self.num_paths})")
        trace = self._traces[path_idx]
        cap = trace.capacity_mbps(self._t, self._rng)
        rtt = trace.rtt_ms(cap)
        loss = _PathTrace.loss(cap, trace.base_mbps)
        # Effective goodput accounts for loss; duration = transfer + one RTT.
        goodput_mbps = max(0.05, cap * (1.0 - loss))
        transfer_s = (segment_bytes * 8.0) / (goodput_mbps * 1e6)
        duration_s = transfer_s + rtt / 1000.0
        self._t += duration_s
        self._segments_done += 1
        return DownloadResult(
            path_idx=path_idx,
            throughput_mbps=goodput_mbps,
            rtt_ms=rtt,
            loss=loss,
            duration_s=duration_s,
            bytes_delivered=int(segment_bytes),
        )

    def is_done(self) -> bool:
        return self._segments_done >= self.config.episode_segments

    # -- internals ---------------------------------------------------------- #

    def _stats_at(self, path_idx: int, t: float) -> PathStats:
        trace = self._traces[path_idx]
        # Observation uses the expected capacity (no fresh noise draw) so that
        # observing does not perturb the realized-download RNG stream.
        season = 1.0 + trace.amp * math.sin(
            2.0 * math.pi * t / trace.period_s + trace.phase
        )
        cap = max(0.05, trace.base_mbps * season)
        return PathStats(
            throughput_mbps=cap,
            rtt_ms=trace.rtt_ms(cap),
            loss=_PathTrace.loss(cap, trace.base_mbps),
        )


# --------------------------------------------------------------------------- #
# NS-3 backend (stub; implemented against ns3-ai in WSL2)
# --------------------------------------------------------------------------- #


class Ns3DataPlane(DataPlane):
    """Drives a real NS-3 scenario via the ns3-ai shared-memory bridge.

    This is a stub: it defines the contract and fails fast off-Linux. The full
    implementation (Phase 1, in WSL2) wires ``current_path_stats`` to the ns3-ai
    observation struct and ``download_segment`` to one decision epoch of the
    ``video_mpquic`` NS-3 scenario (set action -> run sim until the segment is
    delivered -> read result). See ``ns3/README.md``.
    """

    def __init__(self, *, num_paths: int = 3, cap_mbps: float = 10.0, **ns3_kwargs):
        self.num_paths = int(num_paths)
        self.cap_mbps = float(cap_mbps)
        self._ns3_kwargs = ns3_kwargs
        self._bridge = None  # set in reset()

    def _require_bridge(self):
        if self._bridge is None:
            raise RuntimeError(
                "Ns3DataPlane is not yet wired. Build NS-3 + ns3-ai in WSL2 "
                "(see ns3/README.md) and call reset() to start the bridge."
            )
        return self._bridge

    def reset(self, *, seed: Optional[int] = None) -> None:
        try:
            import ns3ai_gym_env  # noqa: F401  (only available in the NS-3 env)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ns3-ai is not importable. The NS-3 backend only runs inside the "
                "WSL2/Linux NS-3 environment; use MockTraceDataPlane elsewhere."
            ) from exc
        raise NotImplementedError(
            "Ns3DataPlane.reset will start the ns3-ai bridge once the NS-3 "
            "scenario in ns3/scratch/video_mpquic.cc is built (Phase 1, WSL2)."
        )

    @property
    def clock_s(self) -> float:  # pragma: no cover - stub
        return float(getattr(self._require_bridge(), "clock_s", 0.0))

    def current_path_stats(self) -> List[PathStats]:  # pragma: no cover - stub
        raise NotImplementedError

    def download_segment(
        self, path_idx: int, segment_bytes: int
    ) -> DownloadResult:  # pragma: no cover - stub
        raise NotImplementedError

    def is_done(self) -> bool:  # pragma: no cover - stub
        raise NotImplementedError
