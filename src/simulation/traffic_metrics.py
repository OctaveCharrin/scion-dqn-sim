"""Link metrics and congestion analysis for traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

EDGE_KEY = Tuple[int, int]


@dataclass(frozen=True)
class LinkMetricArrays:
    """Per-link metrics for one hour (vectorized)."""

    latency_ms: np.ndarray
    available_mbps: np.ndarray
    utilization_capped: np.ndarray
    utilization_raw: np.ndarray
    loss_rate: np.ndarray


def compute_link_metrics_vectorized(
    loads_mbps: np.ndarray,
    capacities_mbps: np.ndarray,
    base_latencies_ms: np.ndarray,
    util_cap: float = 1.5,
) -> LinkMetricArrays:
    """Derive latency, availability, utilization, and loss for all links."""
    caps = np.maximum(capacities_mbps.astype(np.float64), 1.0)
    loads = loads_mbps.astype(np.float64)
    utils_raw = loads / caps
    utils_capped = np.minimum(np.maximum(utils_raw, 0.0), util_cap)
    util_clipped = np.minimum(utils_capped, 0.99)

    denom = np.maximum(0.01, 1.0 - util_clipped)
    queue_mult = 1.0 + 1.0 * (util_clipped / denom) * 0.05
    queue_mult = np.minimum(queue_mult, 2.0)
    latency = base_latencies_ms.astype(np.float64) * queue_mult
    available = np.maximum(0.0, caps - loads)

    loss = np.zeros_like(loads)
    high = utils_capped >= 0.95
    mid = (utils_capped >= 0.8) & ~high
    loss[high] = np.minimum(
        0.2, 0.001 + 0.05 * (utils_capped[high] - 0.95) / 0.05
    )
    loss[mid] = 0.001 + 0.005 * (utils_capped[mid] - 0.8) / 0.15

    return LinkMetricArrays(
        latency_ms=latency,
        available_mbps=available,
        utilization_capped=utils_capped,
        utilization_raw=utils_raw,
        loss_rate=loss,
    )


def path_metrics_from_link_indices(
    link_indices: Sequence[int],
    link_metrics: LinkMetricArrays,
) -> Dict[str, float]:
    """Bottleneck aggregation for one path (min avail, max util, product loss)."""
    if not link_indices:
        return {
            "latency_ms": 0.0,
            "available_bandwidth_mbps": 0.0,
            "utilization": 0.0,
            "utilization_raw": 0.0,
            "loss_rate": 0.0,
            "hop_count": 0,
        }
    idx = np.asarray(link_indices, dtype=np.intp)
    lat = float(link_metrics.latency_ms[idx].sum())
    avail = float(link_metrics.available_mbps[idx].min())
    util = float(link_metrics.utilization_capped[idx].max())
    util_raw = float(link_metrics.utilization_raw[idx].max())
    survive = float(np.prod(1.0 - link_metrics.loss_rate[idx]))
    return {
        "latency_ms": lat,
        "available_bandwidth_mbps": avail,
        "utilization": util,
        "utilization_raw": util_raw,
        "loss_rate": float(1.0 - survive),
        "hop_count": int(len(idx)),
    }


def summarize_link_loads(
    loads_by_hour: Dict[int, np.ndarray],
    capacities_mbps: np.ndarray,
    *,
    peak_hour: int | None = None,
) -> Dict[str, float]:
    """Aggregate utilization stats across hours (raw and capped)."""
    if not loads_by_hour:
        return {}

    caps = np.maximum(capacities_mbps.astype(np.float64), 1.0)
    all_utils_raw: List[float] = []
    all_utils_pos: List[float] = []
    peak_utils_raw: List[float] = []

    for h, loads in loads_by_hour.items():
        utils = loads.astype(np.float64) / caps
        mask = loads > 0
        if mask.any():
            all_utils_pos.extend(utils[mask].tolist())
        all_utils_raw.extend(utils.tolist())
        if peak_hour is None or h == peak_hour:
            peak_utils_raw.extend(utils.tolist())

    def _pct(arr: List[float], q: float) -> float:
        return float(np.percentile(arr, q)) if arr else 0.0

    return {
        "util_raw_mean": float(np.mean(all_utils_raw)) if all_utils_raw else 0.0,
        "util_raw_p50": _pct(all_utils_raw, 50),
        "util_raw_p90": _pct(all_utils_raw, 90),
        "util_raw_max": float(max(all_utils_raw)) if all_utils_raw else 0.0,
        "util_on_loaded_links_p90": _pct(all_utils_pos, 90),
        "util_peak_hour_raw_p90": _pct(peak_utils_raw, 90),
        "util_peak_hour_raw_max": float(max(peak_utils_raw)) if peak_utils_raw else 0.0,
        "fraction_links_util_gt_1": float(
            np.mean(np.array(all_utils_raw) > 1.0)
        )
        if all_utils_raw
        else 0.0,
    }
