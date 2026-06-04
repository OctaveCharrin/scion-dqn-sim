"""Inspect congestion and traffic quality for a completed step-03 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.simulation.link_state_store import (
    build_pair_path_link_idx_for_pool,
    load_link_traffic_state,
)
from src.simulation.path_store import InMemoryPathStore
from src.simulation.run_context import topology_dir


def _peak_diurnal_hour() -> int:
    """Matches ``_diurnal_factor`` peak in link_traffic_sim (~hour 14)."""
    return 14


def analyze_traffic_run(
    run_path: Path,
    *,
    eval_pair_limit: int = 32,
    sample_pairs_for_path_stats: int = 64,
) -> Dict[str, Any]:
    """Build a congestion / path-quality report from ``link_states.pkl``.

    Reads ``simulation_metadata.json`` when present for calibration context.
  Works with compact ``link_hourly_v1`` and legacy per-pair embedded states.
    """
    run_path = Path(run_path)
    meta_path = run_path / "simulation_metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())

    link_state = load_link_traffic_state(run_path / "link_states.pkl")
    path_store = InMemoryPathStore.load(run_path / "path_store.json")

    with open(run_path / "selected_pair.json") as f:
        selected = json.load(f)

    pair_pool = [
        (int(p[0]), int(p[1]))
        for p in selected.get("pair_pool", [])
    ]
    if not pair_pool:
        pair_pool = [
            (int(selected["source_as"]), int(selected["destination_as"]))
        ]
    eval_pairs = pair_pool[: min(len(pair_pool), eval_pair_limit)]

    hours = link_state.hour_keys()
    if not hours:
        raise ValueError("link_states.pkl is empty")

    train_end = 14 * 24
    eval_hours = [h for h in hours if h >= train_end]
    peak_hours = [h for h in hours if (h % 24) == _peak_diurnal_hour()]

    pair_path_link_idx = {}
    if not link_state.is_legacy:
        pair_path_link_idx = build_pair_path_link_idx_for_pool(
            path_store,
            pair_pool,
            link_state.link_key_to_index,
        )

    rng = np.random.default_rng(0)
    if len(pair_pool) > sample_pairs_for_path_stats:
        idx = rng.choice(len(pair_pool), sample_pairs_for_path_stats, replace=False)
        sample_pairs = [pair_pool[int(i)] for i in idx]
    else:
        sample_pairs = pair_pool

    zero_bw = 0
    total_ph = 0
    max_bws: List[float] = []
    path_utils_raw: List[float] = []

    for h in eval_hours:
        for pair in sample_pairs:
            per_pair = link_state.path_metrics_block(
                h, int(pair[0]), int(pair[1]), pair_path_link_idx
            )
            if not per_pair:
                continue
            bws = [
                float(v.get("available_bandwidth_mbps", 0.0)) for v in per_pair.values()
            ]
            utils = [float(v.get("utilization_raw", 0.0)) for v in per_pair.values()]
            if not bws:
                continue
            total_ph += 1
            max_bws.append(max(bws))
            path_utils_raw.extend(utils)
            if max(bws) <= 0.001:
                zero_bw += 1

    link_utils_peak: List[float] = []
    if not link_state.is_legacy and peak_hours:
        for h in peak_hours:
            lm = link_state.link_metrics_at(h)
            link_utils_peak.extend(lm.utilization_raw.tolist())

    report: Dict[str, Any] = {
        "run_dir": str(run_path),
        "link_state_format": (
            "legacy_by_pair" if link_state.is_legacy else "link_hourly_v1"
        ),
        "num_pairs_in_pool": len(pair_pool),
        "num_links": len(link_state.link_keys),
        "hours_total": len(hours),
        "eval_hours": len(eval_hours),
        "path_quality": {
            "fraction_max_path_bw_zero": float(zero_bw / total_ph) if total_ph else 0.0,
            "pair_hours_sampled": total_ph,
            "max_path_bw_mbps": {
                "mean": float(np.mean(max_bws)) if max_bws else 0.0,
                "p50": float(np.percentile(max_bws, 50)) if max_bws else 0.0,
                "p90": float(np.percentile(max_bws, 90)) if max_bws else 0.0,
            },
            "path_utilization_raw": {
                "p90": float(np.percentile(path_utils_raw, 90)) if path_utils_raw else 0.0,
                "max": float(np.max(path_utils_raw)) if path_utils_raw else 0.0,
            },
        },
        "link_utilization_peak_diurnal": {
            "p90": float(np.percentile(link_utils_peak, 90)) if link_utils_peak else 0.0,
            "max": float(np.max(link_utils_peak)) if link_utils_peak else 0.0,
        },
        "eval_pairs_sample": [list(p) for p in eval_pairs[:8]],
    }
    if meta:
        report["traffic_calibration"] = meta.get("traffic_calibration", meta)
        report["link_utilization_from_sim"] = meta.get("link_utilization", {})
    return report


def format_traffic_report(report: Dict[str, Any]) -> str:
    """Human-readable summary for CLI output."""
    pq = report.get("path_quality", {})
    lines = [
        f"Run: {report.get('run_dir', '?')}",
        f"Format: {report.get('link_state_format', '?')}",
        f"Pairs in pool: {report.get('num_pairs_in_pool', '?')}",
        f"Links: {report.get('num_links', '?')}",
        f"Hours: {report.get('hours_total', '?')} (eval window: {report.get('eval_hours', '?')})",
        "",
        "Path quality (sampled pairs × eval hours):",
        f"  Zero max-path-BW fraction: {pq.get('fraction_max_path_bw_zero', 0):.1%}",
        f"  Pair-hours sampled: {pq.get('pair_hours_sampled', 0)}",
    ]
    mbw = pq.get("max_path_bw_mbps", {})
    if mbw:
        lines.append(
            f"  Max path BW (Mbps): mean={mbw.get('mean', 0):.1f} "
            f"p50={mbw.get('p50', 0):.1f} p90={mbw.get('p90', 0):.1f}"
        )
    util = report.get("link_utilization_from_sim", {})
    if util:
        lines.extend(
            [
                "",
                "Link utilization (from simulation_metadata):",
                f"  Peak-hour raw p90: {util.get('util_peak_hour_raw_p90', 0):.3f}",
                f"  Peak-hour raw max: {util.get('util_peak_hour_raw_max', 0):.3f}",
            ]
        )
    return "\n".join(lines)
