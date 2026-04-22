"""
Minimal path-selection environment for ``evaluation/04_train_dqn.py`` and
``05_evaluate_methods.py`` using precomputed paths and hourly link states.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

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


class EvaluationPathSelectionEnv:
    """
    Not a full Gym env: only the methods used by the evaluation scripts.

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
        probe_type: str = "adaptive",
    ) -> None:
        self.topology_data = topology_data
        self.path_store = path_store
        self.link_states = link_states
        self.latency_probe_cost_ms = latency_probe_cost_ms
        self.bandwidth_probe_cost_ms = bandwidth_probe_cost_ms
        self.probe_type = probe_type

        self.current_link_states: Dict[str, Any] = {}
        self.available_paths: List[Any] = []
        self.probed_path_metrics: Dict[int, Dict[str, Any]] = {}
        self.current_flow: Dict[str, Any] = {}
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.current_step = 0

    def reset(
        self,
        source_as: Optional[int] = None,
        dest_as: Optional[int] = None,
    ) -> np.ndarray:
        assert source_as is not None and dest_as is not None
        raw_paths = self.path_store.find_paths(int(source_as), int(dest_as))
        self.available_paths = [_wrap_path(p) if isinstance(p, dict) else p for p in raw_paths]
        self.probed_path_metrics.clear()
        self.current_flow = {"src": int(source_as), "dst": int(dest_as)}
        self.current_step = 0
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        return np.zeros(5, dtype=np.float32)

    def _static_metrics(self, path_idx: int) -> Dict[str, Any]:
        p = self.available_paths[path_idx]
        if isinstance(p, SimpleNamespace):
            sm = getattr(p, "static_metrics", None) or {}
        elif isinstance(p, dict):
            sm = p.get("static_metrics", {})
        else:
            sm = {}
        return dict(sm)

    def probe_path_latency(self, path_index: int) -> Dict[str, Any]:
        if path_index >= len(self.available_paths):
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": None,
                "loss_rate": 0.01,
                "hop_count": 0,
                "probe_type": "latency",
            }
        sm = self._static_metrics(path_index)
        lat = float(sm.get("total_latency", 50.0)) + self.latency_probe_cost_ms
        hop = int(sm.get("hop_count", 1))
        self.num_latency_probes += 1
        return {
            "latency_ms": lat,
            "bandwidth_mbps": None,
            "loss_rate": 0.01,
            "hop_count": hop,
            "probe_type": "latency",
        }

    def probe_path_full(self, path_index: int) -> Dict[str, Any]:
        if path_index >= len(self.available_paths):
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": 0.0,
                "loss_rate": 1.0,
                "hop_count": 0,
                "probe_type": "full",
            }
        sm = self._static_metrics(path_index)
        lat = float(sm.get("total_latency", 50.0)) + self.latency_probe_cost_ms
        bw = float(sm.get("min_bandwidth", 1000.0))
        hop = int(sm.get("hop_count", 1))
        self.num_latency_probes += 1
        self.num_bandwidth_probes += 1
        out = {
            "latency_ms": lat,
            "bandwidth_mbps": bw,
            "loss_rate": 0.01,
            "hop_count": hop,
            "probe_type": "full",
        }
        self.probed_path_metrics[path_index] = out
        return out

    def step(self, action: int) -> tuple:
        self.current_step += 1
        path_metrics: Dict[str, Any]
        if action < len(self.available_paths):
            st = self.current_link_states.get(f"path_{action}", {})
            sm = self._static_metrics(action)
            path_metrics = {
                "latency_ms": float(st.get("latency_ms", sm.get("total_latency", 50.0))),
                "bandwidth_mbps": float(
                    st.get("available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0))
                ),
                "loss_rate": float(st.get("loss_rate", 0.0)),
                "hop_count": int(sm.get("hop_count", 1)),
            }
        else:
            path_metrics = {
                "latency_ms": float("inf"),
                "bandwidth_mbps": 0.0,
                "loss_rate": 1.0,
                "hop_count": 0,
            }
        info = {
            "path_metrics": path_metrics,
            "probe_count": self.num_latency_probes + self.num_bandwidth_probes,
        }
        return np.zeros(5, dtype=np.float32), 0.0, False, info
