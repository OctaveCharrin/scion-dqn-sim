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
* :class:`Ns3DataPlane` -- drives a real NS-3 packet-level scenario over the
  ns3-ai shared-memory bridge (Linux/WSL2). One long-lived NS-3 process serves
  the whole run; episode boundaries are sent in-band. The method contract is
  identical to the mock, so the env code is backend-agnostic.

Design note: keeping *all* network modelling behind this interface is what lets
RL development proceed independently of the C++ / ns3-ai integration risk.
"""

from __future__ import annotations

import gc
import glob
import math
import os
import sys
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


@dataclass
class Ns3Config:
    """Episode parameters for :class:`Ns3DataPlane`.

    Mirrors the fields :class:`~src.ns3env.video_env.Ns3VideoMpquicEnv` reads off
    ``dataplane.config`` (``episode_segments`` for progress, ``segment_bytes``
    for the default per-step size). They must match the NS-3 scenario's CLI
    arguments, which are forwarded verbatim by :meth:`Ns3DataPlane.reset`.
    """

    episode_segments: int = 48
    segment_bytes: int = 500_000
    seed: int = 1


@dataclass
class _EnvSnapshot:
    """Pure-Python copy of one ``EnvStruct`` read across the bridge.

    The shared-memory struct is only valid between ``PyRecvBegin``/``PyRecvEnd``;
    we snapshot the fields we need so the rest of the code can read them freely.
    """

    num_paths: int
    clock_s: float
    done: bool
    last_path: int
    last_throughput_mbps: float
    last_rtt_ms: float
    last_loss: float
    last_duration_s: float
    last_bytes: int
    throughput: List[float]
    rtt: List[float]
    loss: List[float]


# Default location of the built NS-3 tree (overridable via $NS3_DIR or the
# ``ns3_dir`` kwarg). The video-mpquic example lives under contrib/ai.
_DEFAULT_NS3_DIR = os.path.expanduser("~/ns-3-dev")
_EXAMPLE_SUBPATH = os.path.join("contrib", "ai", "examples", "video-mpquic")
_NS3_TARGET = "ns3ai_video_mpquic"
_PY_MODULE = "ns3ai_video_mpquic_py"

# Action commands; must match enum ActCommand in video_mpquic.h.
_CMD_STEP = 0
_CMD_RESET = 1
_CMD_TERMINATE = 2


class Ns3DataPlane(DataPlane):
    """Drives the real NS-3 ``video_mpquic`` scenario via the ns3-ai bridge.

    Python is the parent process: :meth:`reset` launches the NS-3 binary through
    ``ns3ai_utils.Experiment`` and then drives the per-segment decision loop over
    shared memory. The C++ controller leads every decision with a send, so:

    * ``reset``          → start the process, receive the initial observation.
    * ``download_segment`` → send the action, receive the next observation, whose
      ``last*`` fields carry the realized result of the segment just delivered.

    The struct fields (``contrib/ai/examples/video-mpquic/video_mpquic.h``)
    mirror :class:`PathStats` / :class:`DownloadResult`. See ``ns3/README.md``.

    ns3-ai allows only **one** shared-memory creator per Python process, so a
    single long-lived NS-3 process serves the whole run: the first :meth:`reset`
    launches it, and every later :meth:`reset` starts a new episode **in-band**
    (``ACT_RESET``) while the simulation keeps running. Use one ``Ns3DataPlane``
    per process; :meth:`close` ends the process and frees the shared memory.
    """

    def __init__(
        self,
        *,
        num_paths: int = 3,
        cap_mbps: float = 10.0,
        ns3_dir: Optional[str] = None,
        config: Optional[Ns3Config] = None,
        show_output: bool = False,
    ):
        self.num_paths = int(num_paths)
        self.cap_mbps = float(cap_mbps)
        self.config = config or Ns3Config()
        self.show_output = bool(show_output)
        self.ns3_dir = os.path.abspath(
            ns3_dir or os.environ.get("NS3_DIR", _DEFAULT_NS3_DIR)
        )

        self._binding = None  # imported pybind module
        self._exp = None  # ns3ai_utils.Experiment
        self._msg = None  # Ns3AiMsgInterfaceImpl
        self._env: Optional[_EnvSnapshot] = None
        self._finished = False
        self._started = False  # whether the NS-3 process is live
        # After reset()/download_segment() the C++ side is blocked waiting for an
        # action, i.e. Python owes exactly one send (used by close()).
        self._owe_send = False

    # -- lifecycle ---------------------------------------------------------- #

    def _import_binding(self):
        if self._binding is not None:
            return self._binding
        example_dir = os.path.join(self.ns3_dir, _EXAMPLE_SUBPATH)
        so_glob = os.path.join(example_dir, f"{_PY_MODULE}*.so")
        if not glob.glob(so_glob):
            raise RuntimeError(
                f"ns3-ai binding {_PY_MODULE} not found in {example_dir}. Build "
                f"it with `./ns3 build {_NS3_TARGET}` in {self.ns3_dir} "
                "(see ns3/README.md)."
            )
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)
        import importlib

        try:
            self._binding = importlib.import_module(_PY_MODULE)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Failed to import {_PY_MODULE} from {example_dir}; the NS-3 "
                "backend only runs inside the WSL2/Linux NS-3 environment."
            ) from exc
        return self._binding

    def _teardown(self) -> None:
        if self._exp is not None:
            try:
                self._exp.kill()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._exp = None
            self._msg = None
            gc.collect()  # let Experiment.__del__ release the shared memory

    def _launch(self) -> None:
        """Start the NS-3 process (once per data plane) and attach the bridge.

        ns3-ai permits only one shared-memory creator per Python process, so we
        launch a single long-lived NS-3 process and reset episodes in-band.
        """
        binding = self._import_binding()
        # ns3ai_utils launches `./ns3` as a subprocess; make sure it runs under
        # the same (venv) interpreter rather than a stray system python3.
        venv_bin = os.path.dirname(sys.executable)
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

        from ns3ai_utils import Experiment  # noqa: WPS433 (NS-3 env only)

        cwd = os.getcwd()
        try:
            self._exp = Experiment(
                _NS3_TARGET, self.ns3_dir, binding, handleFinish=True
            )
            setting = {
                "segments": self.config.episode_segments,
                "segmentBytes": self.config.segment_bytes,
                "seed": max(1, int(self.config.seed)),  # NS-3 rejects seed 0
            }
            self._msg = self._exp.run(setting=setting, show_output=self.show_output)
        finally:
            os.chdir(cwd)  # Experiment chdir's into ns3_dir; restore caller's cwd
        self._started = True
        self._finished = False

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            # The seed parameterizes the NS-3 process; it only takes effect on
            # the first reset (process launch). Subsequent episodes continue the
            # same evolving simulation. NS-3's RngSeedManager rejects seed 0
            # ("invalid Seed 0"), so clamp into its valid range (>= 1); callers
            # routinely pass seed=base+episode with base 0.
            self.config.seed = max(1, int(seed))

        if not self._started:
            self._launch()
            self._env = None
            self._recv_env()  # initial observation (last_path == -1)
        else:
            # Continuing process: ask C++ to start a fresh episode in-band.
            self._send_act(_CMD_RESET)
            self._recv_env()

        if self._env is not None:
            self.num_paths = self._env.num_paths

    # -- bridge primitives -------------------------------------------------- #

    def _recv_env(self) -> None:
        msg = self._msg
        msg.PyRecvBegin()
        if msg.PyGetFinished():
            self._finished = True
            self._owe_send = False
            msg.PyRecvEnd()
            return
        e = msg.GetCpp2PyStruct()
        n = int(e.numPaths)
        self._env = _EnvSnapshot(
            num_paths=n,
            clock_s=float(e.clockS),
            done=bool(e.done),
            last_path=int(e.lastPath),
            last_throughput_mbps=float(e.lastThroughputMbps),
            last_rtt_ms=float(e.lastRttMs),
            last_loss=float(e.lastLoss),
            last_duration_s=float(e.lastDurationS),
            last_bytes=int(e.lastBytes),
            throughput=[float(e.throughput(i)) for i in range(n)],
            rtt=[float(e.rtt(i)) for i in range(n)],
            loss=[float(e.loss(i)) for i in range(n)],
        )
        msg.PyRecvEnd()
        self._owe_send = True  # C++ is now blocked waiting for our action

    def _send_act(
        self, command: int, path_idx: int = 0, segment_bytes: int = 0
    ) -> None:
        msg = self._msg
        msg.PySendBegin()
        a = msg.GetPy2CppStruct()
        a.command = int(command)
        a.pathIdx = int(path_idx)
        a.segmentBytes = int(segment_bytes)
        msg.PySendEnd()
        self._owe_send = False

    # -- DataPlane API ------------------------------------------------------ #

    @property
    def clock_s(self) -> float:
        return float(self._env.clock_s) if self._env is not None else 0.0

    def current_path_stats(self) -> List[PathStats]:
        if self._env is None:
            raise RuntimeError("Ns3DataPlane.reset() must be called before use")
        env = self._env
        return [
            PathStats(
                throughput_mbps=env.throughput[i],
                rtt_ms=env.rtt[i],
                loss=env.loss[i],
            )
            for i in range(env.num_paths)
        ]

    def download_segment(self, path_idx: int, segment_bytes: int) -> DownloadResult:
        if self._env is None:
            raise RuntimeError("Ns3DataPlane.reset() must be called before use")
        if self.is_done():
            raise RuntimeError("episode already finished; call reset()")
        if not 0 <= path_idx < self.num_paths:
            raise IndexError(f"path_idx {path_idx} out of range [0, {self.num_paths})")
        self._send_act(_CMD_STEP, path_idx, segment_bytes)
        self._recv_env()  # next observation carries this segment's result
        env = self._env
        if env is None:  # process finished unexpectedly
            raise RuntimeError("NS-3 process ended before reporting the segment")
        return DownloadResult(
            path_idx=path_idx,
            throughput_mbps=env.last_throughput_mbps,
            rtt_ms=env.last_rtt_ms,
            loss=env.last_loss,
            duration_s=env.last_duration_s,
            bytes_delivered=int(env.last_bytes),
        )

    def is_done(self) -> bool:
        if self._finished:
            return True
        return bool(self._env.done) if self._env is not None else False

    def close(self) -> None:
        """End the NS-3 process gracefully and release the shared memory.

        Sends ACT_TERMINATE if the C++ side is still waiting for an action, then
        tears down the subprocess. Safe to call repeatedly.
        """
        if self._started and not self._finished and self._owe_send:
            try:
                self._send_act(_CMD_TERMINATE)
            except Exception:  # pragma: no cover - process may already be gone
                pass
        self._teardown()
        self._env = None
        self._finished = False
        self._started = False
        self._owe_send = False

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self._teardown()
