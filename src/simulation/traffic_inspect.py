"""Inspect congestion and traffic quality for a completed step-03 run."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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
    """
    run_path = Path(run_path)
    meta_path = run_path / "simulation_metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())

    with open(run_path / "link_states.pkl", "rb") as f:
        link_states = pickle.load(f)

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

    hours = sorted(link_states.keys())
    if not hours:
        raise ValueError("link_states.pkl is empty")

    train_end = 14 * 24
    eval_hours = [h for h in hours if h >= train_end]
    peak_hours = [h for h in hours if (h % 24) == _peak_diurnal_hour()]

    # Path-level stats on a subset of pairs (full pool can be huge).
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
        block = link_states[h].get("by_pair", {})
        for pair in sample_pairs:
            key = f"pair_{pair[0]}_{pair[1]}"
            per_pair = block.get(key, {})
            if not per_pair:
                continue
            bws = [
                float(v.get("available_bandwidth_mbps", 0.0))
                for v in per_pair.values()
            ]
            if not bws:
                continue
            total_ph += 1
            mx = max(bws)
            max_bws.append(mx)
            if mx <= 0.001:
                zero_bw += 1
            for v in per_pair.values():
                ur = v.get("utilization_raw")
                if ur is not None:
                    path_utils_raw.append(float(ur))

    link_diag = meta.get("link_utilization", {})
    calibration = meta.get("traffic_calibration", {})

    report: Dict[str, Any] = {
        "run_dir": str(run_path),
        "num_hours": len(hours),
        "eval_hours": len(eval_hours),
        "pair_pool_size": len(pair_pool),
        "sample_pairs_for_path_stats": len(sample_pairs),
        "path_quality_eval_window": {
            "pair_hours_sampled": total_ph,
            "fraction_max_path_bw_zero": float(zero_bw / total_ph) if total_ph else 0.0,
            "max_path_bw_mean_mbps": float(np.mean(max_bws)) if max_bws else 0.0,
            "max_path_bw_p50_mbps": float(np.percentile(max_bws, 50)) if max_bws else 0.0,
            "path_util_raw_p90": float(np.percentile(path_utils_raw, 90))
            if path_utils_raw
            else 0.0,
        },
        "link_utilization_from_simulation": link_diag,
        "traffic_calibration": calibration,
        "targets": {
            "p90_utilization": calibration.get("target_p90_utilization", 0.75),
            "max_zero_bw_pair_hour_fraction": calibration.get(
                "target_zero_bw_pair_hour_fraction", 0.20
            ),
        },
        "eval_pairs_count": len(eval_pairs),
    }

    if link_diag:
        p90 = float(link_diag.get("util_peak_hour_raw_p90", 0.0))
        report["calibration_check"] = {
            "p90_util_in_band_0.6_0.9": 0.6 <= p90 <= 0.9,
            "zero_bw_fraction_below_0.3": report["path_quality_eval_window"][
                "fraction_max_path_bw_zero"
            ]
            < 0.30,
        }

    return report


def format_traffic_report(report: Dict[str, Any]) -> str:
    """Human-readable summary for CLI / notebooks."""
    lines = [
        f"Traffic inspection: {report.get('run_dir', '')}",
        f"  Pair pool size: {report.get('pair_pool_size')}",
        f"  Eval hours: {report.get('eval_hours')}",
    ]
    pq = report.get("path_quality_eval_window", {})
    lines.append(
        f"  Fraction (pair,hour) with max path BW == 0 (sample): "
        f"{pq.get('fraction_max_path_bw_zero', 0):.1%}"
    )
    lines.append(
        f"  Mean max-path BW (Mbps): {pq.get('max_path_bw_mean_mbps', 0):.1f}"
    )
    lu = report.get("link_utilization_from_simulation", {})
    if lu:
        lines.append(
            f"  Peak-hour link util raw p90: {lu.get('util_peak_hour_raw_p90', 0):.3f}"
        )
        lines.append(
            f"  Peak-hour link util raw max: {lu.get('util_peak_hour_raw_max', 0):.3f}"
        )
        lines.append(
            f"  Fraction links util>1 (any hour): {lu.get('fraction_links_util_gt_1', 0):.1%}"
        )
    cal = report.get("calibration_check")
    if cal:
        lines.append(f"  Calibration p90 in [0.6,0.9]: {cal.get('p90_util_in_band_0.6_0.9')}")
        lines.append(
            f"  Zero-BW fraction < 30%: {cal.get('zero_bw_fraction_below_0.3')}"
        )
    tc = report.get("traffic_calibration", {})
    if tc:
        lines.append(
            f"  Active pairs/hour: {tc.get('active_pairs_min')}–{tc.get('active_pairs_max')}"
        )
        lines.append(
            f"  Scaled base rate (Mbps): {tc.get('scaled_base_rate_mbps', 0):.2f}"
        )
        lines.append(
            f"  Background pairs/hour: {tc.get('background_pairs_per_hour_typical')}"
        )
    return "\n".join(lines)
