"""
Path-selection environment used by ``evaluation/04_train_dqn.py`` and
``05_evaluate_methods.py``.

The environment now exposes:

* **Selective probing** with separate per-call probe-time accounting.
  ``probe_path_latency`` and ``probe_path_full`` return the *actual* measured
  metric (latency in ms, bottleneck bandwidth in Mbps), and the wall-clock
  cost of issuing the probe is tracked in ``last_probe_cost_ms`` and the
  cumulative ``total_probe_cost_ms``. This used to be conflated by adding
  the probe cost into the returned latency, biasing every reading upward
  by ~10 ms.
* **Stateful episodes** keyed on ``(src, dst, hour_idx)``. ``reset`` may
  pick a fresh AS pair from the path store and select an initial hour; the
  default ``step`` advances ``hour_idx`` by one within an episode of
  configurable length so γ in the DQN bootstrap actually has work to do.
* **Action masks** that mirror ``len(available_paths) <= action_dim``.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _wrap_path(path_dict: Dict[str, Any]) -> SimpleNamespace:
    """Attach ``as_sequence`` for baseline selectors that expect path objects."""
    hops = path_dict.get("hops") or []
    seq = tuple(int(h["as"]) for h in hops if isinstance(h, dict) and "as" in h)
    ns = SimpleNamespace()
    for k, v in path_dict.items():
        setattr(ns, k, v)
    ns.as_sequence = seq
    return ns


def _path_to_link_keys(path: Any) -> List[Tuple[int, int]]:
    """Return canonical ``(min, max)`` link keys for every consecutive hop."""
    hops = getattr(path, "hops", None)
    if hops is None and isinstance(path, dict):
        hops = path.get("hops") or []
    hops = hops or []
    out: List[Tuple[int, int]] = []
    for i in range(len(hops) - 1):
        a, b = int(hops[i]["as"]), int(hops[i + 1]["as"])
        out.append((a, b) if a <= b else (b, a))
    return out


class EvaluationPathSelectionEnv:
    """Path-selection environment with selective probing and stateful episodes.

    ``available_paths`` entries are ``SimpleNamespace`` views with ``hops``,
    ``static_metrics``, and ``as_sequence``.
    """

    def __init__(
        self,
        topology_data: Dict[str, Any],
        path_store: Any,
        link_states: Dict[int, Dict[str, Any]],
        latency_probe_cost_ms: float = 10.0,
        bandwidth_probe_cost_ms: float = 100.0,
        per_hop_probe_cost_ms: float = 0.5,
        per_hop_full_probe_cost_ms: float = 20.0,
        probe_type: str = "adaptive",
        pair_pool: Optional[Sequence[Tuple[int, int]]] = None,
        episode_length: int = 24,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.topology_data = topology_data
        self.path_store = path_store
        self.link_states = link_states
        self.latency_probe_cost_ms = float(latency_probe_cost_ms)
        self.bandwidth_probe_cost_ms = float(bandwidth_probe_cost_ms)
        self.per_hop_probe_cost_ms = float(per_hop_probe_cost_ms)
        self.per_hop_full_probe_cost_ms = float(per_hop_full_probe_cost_ms)
        self.probe_type = probe_type
        self.pair_pool: List[Tuple[int, int]] = (
            [(int(s), int(d)) for (s, d) in pair_pool] if pair_pool else []
        )
        self.episode_length = max(1, int(episode_length))
        self._rng = random.Random(rng_seed)

        # Episode state.
        self.current_link_states: Dict[str, Any] = {}
        self.available_paths: List[Any] = []
        self.probed_path_metrics: Dict[int, Dict[str, Any]] = {}
        self.current_flow: Dict[str, Any] = {}
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.last_probe_cost_ms = 0.0
        self.total_probe_cost_ms = 0.0
        self.current_step = 0
        self.hour_idx = 0
        self.episode_start_hour = 0

    # ------------------------------------------------------------------ paths
    def _static_metrics(self, path_idx: int) -> Dict[str, Any]:
        p = self.available_paths[path_idx]
        if isinstance(p, SimpleNamespace):
            sm = getattr(p, "static_metrics", None) or {}
        elif isinstance(p, dict):
            sm = p.get("static_metrics", {})
        else:
            sm = {}
        return dict(sm)

    def _path_link_keys(self, path_idx: int) -> List[Tuple[int, int]]:
        if path_idx >= len(self.available_paths):
            return []
        return _path_to_link_keys(self.available_paths[path_idx])

    # ---------------------------------------------------------------- episode
    def reset(
        self,
        source_as: Optional[int] = None,
        dest_as: Optional[int] = None,
        *,
        hour_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Start a new episode.

        If ``source_as`` / ``dest_as`` are omitted and a non-empty ``pair_pool``
        was provided, a random pair from the pool is chosen.
        """
        if source_as is None or dest_as is None:
            if not self.pair_pool:
                raise ValueError(
                    "EvaluationPathSelectionEnv.reset: no source/dest provided "
                    "and pair_pool is empty."
                )
            source_as, dest_as = self._rng.choice(self.pair_pool)

        raw_paths = self.path_store.find_paths(int(source_as), int(dest_as))
        self.available_paths = [
            _wrap_path(p) if isinstance(p, dict) else p for p in raw_paths
        ]
        self.probed_path_metrics.clear()
        self.current_flow = {"src": int(source_as), "dst": int(dest_as)}
        self.current_step = 0
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.last_probe_cost_ms = 0.0
        self.total_probe_cost_ms = 0.0

        if hour_idx is None:
            if self.link_states:
                hour_idx = self._rng.choice(sorted(self.link_states.keys()))
            else:
                hour_idx = 0
        self.hour_idx = int(hour_idx)
        self.episode_start_hour = int(hour_idx)
        self._refresh_link_states()

        return np.zeros(5, dtype=np.float32)

    def _refresh_link_states(self) -> None:
        """Resolve hourly link states for the current ``(src, dst)`` pair.

        Supports two storage layouts:

        * **Multi-pair** (preferred):
          ``link_states[hour]["by_pair"]["pair_<src>_<dst>"]["path_<idx>"]``
        * **Legacy single-pair**:
          ``link_states[hour]["path_<idx>"]`` — used only when ``by_pair`` is
          absent or doesn't include the current pair.
        """
        hour = self.link_states.get(self.hour_idx, {}) or {}
        src = self.current_flow.get("src")
        dst = self.current_flow.get("dst")
        per_pair = (hour.get("by_pair") or {}) if isinstance(hour, dict) else {}
        key = f"pair_{int(src)}_{int(dst)}" if src is not None and dst is not None else None
        block = per_pair.get(key) if key else None

        if block:
            self.current_link_states = dict(block)
        else:
            # Strip the multi-pair sub-table to keep ``path_<idx>`` keys at the
            # top level for legacy callers that pre-set ``current_link_states``.
            cleaned = {
                k: v for k, v in hour.items() if isinstance(k, str) and k.startswith("path_")
            }
            self.current_link_states = cleaned

    # ------------------------------------------------------------------ probe
    def probe_path_latency(self, path_index: int) -> Dict[str, Any]:
        """Probe a path for latency. ``last_probe_cost_ms`` is updated; the
        returned ``latency_ms`` is the *measured path latency* and does NOT
        include the probe overhead (selection time accounting is separate).
        """
        if path_index >= len(self.available_paths):
            self.last_probe_cost_ms = self.latency_probe_cost_ms
            self.total_probe_cost_ms += self.last_probe_cost_ms
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": None,
                "loss_rate": 1.0,
                "hop_count": 0,
                "probe_type": "latency",
                "probe_cost_ms": self.last_probe_cost_ms,
            }

        sm = self._static_metrics(path_index)
        st = self.current_link_states.get(f"path_{path_index}", {}) or {}
        lat = float(st.get("latency_ms", sm.get("total_latency", 50.0)))
        loss = float(st.get("loss_rate", 0.0))
        hop = int(sm.get("hop_count", 1))
        cost = self.latency_probe_cost_ms + self.per_hop_probe_cost_ms * hop

        self.num_latency_probes += 1
        self.last_probe_cost_ms = cost
        self.total_probe_cost_ms += cost

        return {
            "latency_ms": lat,
            "bandwidth_mbps": None,
            "loss_rate": loss,
            "hop_count": hop,
            "probe_type": "latency",
            "probe_cost_ms": cost,
        }

    def probe_path_full(self, path_index: int) -> Dict[str, Any]:
        """Probe a path for both latency and bandwidth."""
        if path_index >= len(self.available_paths):
            self.last_probe_cost_ms = self.bandwidth_probe_cost_ms
            self.total_probe_cost_ms += self.last_probe_cost_ms
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": 0.0,
                "loss_rate": 1.0,
                "hop_count": 0,
                "probe_type": "full",
                "probe_cost_ms": self.last_probe_cost_ms,
            }

        sm = self._static_metrics(path_index)
        st = self.current_link_states.get(f"path_{path_index}", {}) or {}
        lat = float(st.get("latency_ms", sm.get("total_latency", 50.0)))
        bw = float(
            st.get("available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0))
        )
        loss = float(st.get("loss_rate", 0.0))
        hop = int(sm.get("hop_count", 1))
        cost = (
            self.bandwidth_probe_cost_ms
            + self.per_hop_full_probe_cost_ms * hop
        )

        self.num_latency_probes += 1
        self.num_bandwidth_probes += 1
        self.last_probe_cost_ms = cost
        self.total_probe_cost_ms += cost

        out = {
            "latency_ms": lat,
            "bandwidth_mbps": bw,
            "loss_rate": loss,
            "hop_count": hop,
            "probe_type": "full",
            "probe_cost_ms": cost,
        }
        self.probed_path_metrics[path_index] = out
        return out

    # ------------------------------------------------------------------ step
    def step(self, action: int) -> tuple:
        """Apply ``action``, advance to the next hour, return (obs, r, done, info).

        Reward is left at 0.0 — training scripts compute it from the path
        metrics info dict to keep the reward weights configurable. The
        observation is a 5-vector zero placeholder; training scripts replace
        it with their own state featurization.
        """
        self.current_step += 1
        path_metrics: Dict[str, Any]
        if action < len(self.available_paths):
            st = self.current_link_states.get(f"path_{action}", {}) or {}
            sm = self._static_metrics(action)
            path_metrics = {
                "latency_ms": float(
                    st.get("latency_ms", sm.get("total_latency", 50.0))
                ),
                "bandwidth_mbps": float(
                    st.get(
                        "available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0)
                    )
                ),
                "loss_rate": float(st.get("loss_rate", 0.0)),
                "hop_count": int(sm.get("hop_count", 1)),
                "utilization": float(st.get("utilization", 0.0)),
            }
        else:
            path_metrics = {
                "latency_ms": float("inf"),
                "bandwidth_mbps": 0.0,
                "loss_rate": 1.0,
                "hop_count": 0,
                "utilization": 1.0,
            }

        # Advance to the next hour and roll over within the episode.
        keys = sorted(self.link_states.keys()) if self.link_states else []
        if keys:
            self.hour_idx = (self.hour_idx + 1) % (max(keys) + 1)
        self._refresh_link_states()

        done = self.current_step >= self.episode_length
        info = {
            "path_metrics": path_metrics,
            "probe_count": self.num_latency_probes + self.num_bandwidth_probes,
            "probe_cost_ms": self.total_probe_cost_ms,
            "hour_idx": self.hour_idx,
        }
        return np.zeros(5, dtype=np.float32), 0.0, done, info

    # ----------------------------------------------------------- introspection
    def num_paths(self) -> int:
        return len(self.available_paths)

    def action_mask(self, action_dim: int) -> np.ndarray:
        """Boolean mask for valid actions given the current pair's path count."""
        n = len(self.available_paths)
        mask = np.zeros(int(action_dim), dtype=bool)
        mask[: min(n, int(action_dim))] = True
        return mask
