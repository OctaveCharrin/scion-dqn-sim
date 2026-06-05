"""Compact hourly link-level traffic state (path metrics derived on demand)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.simulation.traffic_metrics import (
    LinkMetricArrays,
    path_metrics_for_pair,
    path_metrics_from_link_indices,
)

LINK_STATE_FORMAT = "link_hourly_v1"
EDGE_KEY = Tuple[int, int]

_ARRAY_FIELDS = (
    "latency_ms",
    "available_mbps",
    "utilization",
    "utilization_raw",
    "loss_rate",
)


def link_metrics_to_arrays(lm: LinkMetricArrays) -> Dict[str, np.ndarray]:
    return {
        "latency_ms": np.asarray(lm.latency_ms, dtype=np.float64),
        "available_mbps": np.asarray(lm.available_mbps, dtype=np.float64),
        "utilization": np.asarray(lm.utilization_capped, dtype=np.float64),
        "utilization_raw": np.asarray(lm.utilization_raw, dtype=np.float64),
        "loss_rate": np.asarray(lm.loss_rate, dtype=np.float64),
    }


def arrays_to_link_metrics(arrays: Mapping[str, np.ndarray]) -> LinkMetricArrays:
    return LinkMetricArrays(
        latency_ms=arrays["latency_ms"],
        available_mbps=arrays["available_mbps"],
        utilization_capped=arrays["utilization"],
        utilization_raw=arrays["utilization_raw"],
        loss_rate=arrays["loss_rate"],
    )


def pack_hourly_file(
    link_keys: Sequence[EDGE_KEY],
    hours: Mapping[int, Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    return {
        "format": LINK_STATE_FORMAT,
        "link_keys": [[int(a), int(b)] for a, b in link_keys],
        "hours": {int(h): dict(arrays) for h, arrays in hours.items()},
    }


@dataclass
class LinkTrafficState:
    """Hourly link metrics; path-level views built on demand."""

    link_keys: List[EDGE_KEY]
    hours: Dict[int, Dict[str, np.ndarray]]
    legacy_by_hour: Optional[Dict[int, Dict[str, Any]]] = None

    @property
    def is_legacy(self) -> bool:
        return self.legacy_by_hour is not None

    def hour_keys(self) -> List[int]:
        if self.legacy_by_hour is not None:
            return sorted(self.legacy_by_hour.keys())
        return sorted(self.hours.keys())

    @property
    def link_key_to_index(self) -> Dict[EDGE_KEY, int]:
        return {k: i for i, k in enumerate(self.link_keys)}

    def link_metrics_at(self, hour: int) -> LinkMetricArrays:
        if self.legacy_by_hour is not None:
            raise TypeError("link_metrics_at() requires compact link_hourly_v1 format")
        return arrays_to_link_metrics(self.hours[int(hour)])

    def path_metrics_block(
        self,
        hour: int,
        src: int,
        dst: int,
        pair_path_link_idx: Mapping[Tuple[int, int], List[List[int]]],
    ) -> Dict[str, Dict[str, float]]:
        """Return ``path_{i}`` metric dicts for the current (src, dst) pair."""
        if self.legacy_by_hour is not None:
            hour_data = self.legacy_by_hour.get(int(hour), {}) or {}
            per_pair = hour_data.get("by_pair") or {}
            block = per_pair.get(f"pair_{int(src)}_{int(dst)}")
            if block:
                return dict(block)
            return {
                k: v
                for k, v in hour_data.items()
                if isinstance(k, str) and k.startswith("path_")
            }

        lm = self.link_metrics_at(hour)
        keys_list = pair_path_link_idx.get((int(src), int(dst)), [])
        return path_metrics_for_pair(keys_list, lm)


def load_link_traffic_state(path: Path) -> LinkTrafficState:
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and data.get("format") == LINK_STATE_FORMAT:
        link_keys = [(int(a), int(b)) for a, b in data["link_keys"]]
        hours = {int(h): v for h, v in data["hours"].items()}
        return LinkTrafficState(link_keys=link_keys, hours=hours)

    if isinstance(data, dict):
        return LinkTrafficState(link_keys=[], hours={}, legacy_by_hour=data)

    raise TypeError(f"Unsupported link_states.pkl type: {type(data)}")


def link_state_to_json_dict(state: LinkTrafficState) -> Dict[str, Any]:
    """JSON-serializable view of traffic state (arrays as lists)."""
    if state.is_legacy and state.legacy_by_hour is not None:
        return state.legacy_by_hour
    return {
        "format": LINK_STATE_FORMAT,
        "link_keys": [[int(a), int(b)] for a, b in state.link_keys],
        "hours": {
            int(h): {k: np.asarray(v).tolist() for k, v in arrays.items()}
            for h, arrays in state.hours.items()
        },
    }


def save_link_traffic_state(path: Path, state: LinkTrafficState) -> None:
    if state.legacy_by_hour is not None:
        payload: Any = state.legacy_by_hour
    else:
        payload = pack_hourly_file(state.link_keys, state.hours)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_pair_path_link_idx_for_pool(
    path_store: Any,
    pair_pool: Sequence[Tuple[int, int]],
    link_key_to_index: Mapping[EDGE_KEY, int],
) -> Dict[Tuple[int, int], List[List[int]]]:
    from src.simulation.link_traffic_sim import _path_link_keys

    out: Dict[Tuple[int, int], List[List[int]]] = {}
    for pair in pair_pool:
        plist = path_store.find_paths(int(pair[0]), int(pair[1]))
        idx_lists: List[List[int]] = []
        for p in plist:
            keys = _path_link_keys(p)
            idx_lists.append(
                [link_key_to_index[k] for k in keys if k in link_key_to_index]
            )
        out[(int(pair[0]), int(pair[1]))] = idx_lists
    return out
