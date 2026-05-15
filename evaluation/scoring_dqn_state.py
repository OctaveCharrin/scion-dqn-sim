"""Shared dict-state featurization for path-scoring DQN (04 / 05)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

GLOBAL_DIM = 5
PATH_FEATURE_DIM = 6


def _static_metrics_dict(path_obj: Any) -> Dict[str, Any]:
    if hasattr(path_obj, "static_metrics"):
        sm = getattr(path_obj, "static_metrics", None) or {}
    elif isinstance(path_obj, dict):
        sm = path_obj.get("static_metrics") or {}
    else:
        sm = {}
    return dict(sm)


def aggregate_global_state(env: Any, hour_idx: int, w3: float, w4: float) -> np.ndarray:
    """Same 5-D context as the flat DQN scripts (day, hour, utilization, trust, congestion)."""
    day = (hour_idx // 24) % 7
    hour = hour_idx % 24
    f0 = day / 6.0
    f1 = hour / 23.0
    states = list(env.current_link_states.values())
    if not states:
        return np.array([f0, f1, 0.0, 0.0, 0.0], dtype=np.float32)
    utils = [float(s.get("utilization", 0.0)) for s in states]
    losses = [float(s.get("loss_rate", 0.0)) for s in states]
    lats = [min(100.0, float(s.get("latency_ms", 50.0))) / 100.0 for s in states]
    trusts = [
        max(0.0, min(1.0, 1.0 - (w3 * loss + w4 * lat)))
        for loss, lat in zip(losses, lats)
    ]
    f2 = float(np.mean(utils)) if utils else 0.0
    f3 = float(np.mean(trusts)) if trusts else 0.0
    f4 = float(np.mean([1.0 if u > 0.7 else 0.0 for u in utils])) if utils else 0.0
    return np.array([f0, f1, f2, f3, f4], dtype=np.float32)


def path_features_matrix(env: Any) -> np.ndarray:
    """Per-path features (N, PATH_FEATURE_DIM); row i matches action index i."""
    n = len(env.available_paths)
    if n == 0:
        return np.zeros((0, PATH_FEATURE_DIM), dtype=np.float32)
    rows = np.zeros((n, PATH_FEATURE_DIM), dtype=np.float32)
    for path_idx in range(n):
        sm = _static_metrics_dict(env.available_paths[path_idx])
        st = env.current_link_states.get(f"path_{path_idx}", {}) or {}
        lat = float(st.get("latency_ms", sm.get("total_latency", 50.0))) / 100.0
        loss = float(st.get("loss_rate", 0.0))
        hop = float(sm.get("hop_count", 1)) / 20.0
        bw = float(st.get("available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0))) / 10000.0
        util = float(st.get("utilization", 0.0))
        static_bw = float(sm.get("min_bandwidth", 0.0)) / 10000.0
        rows[path_idx] = (lat, loss, hop, bw, util, static_bw)
    return rows


def dict_state(env: Any, hour_idx: int, w3: float, w4: float) -> Dict[str, np.ndarray]:
    return {
        "global": aggregate_global_state(env, hour_idx, w3, w4),
        "paths": path_features_matrix(env),
    }
