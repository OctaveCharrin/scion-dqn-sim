#!/usr/bin/env python3
"""Does path choice actually matter in this environment?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pipeline.intent_cond_eval import EVAL_HOURS, MAX_EVAL_PAIRS, _eval_pairs
from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.evaluation_env import RewardWeights
from src.simulation.run_context import load_run_context, make_env


def _pct(values: Sequence[float], q: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=float), q))
        if len(values)
        else float("nan")
    )


def analyze(
    run_path: Path,
    *,
    max_pairs: int,
    hour_stride: int,
    pairs: Optional[Sequence[Tuple[int, int]]],
    progress=print,
) -> Dict[str, Any]:
    topology_data, path_store, link_states, pair_pool, _cap = load_run_context(run_path)
    eval_pairs = _eval_pairs(pair_pool, max_pairs, pairs)
    hours = EVAL_HOURS[:: max(1, hour_stride)]

    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=RewardWeights(),
    )

    bw_ratio_best_median: List[float] = []
    bw_ratio_best_worst: List[float] = []
    bw_cv: List[float] = []
    bw_spread_abs: List[float] = []
    bw_best: List[float] = []
    lat_spread_abs: List[float] = []
    lat_ratio_worst_best: List[float] = []
    n_paths_list: List[int] = []
    dead_contexts = 0

    # Do the per-intent optima coincide?
    same_bw_lat = 0
    same_bw_loss = 0
    same_lat_loss = 0
    all_three_same = 0
    n_ctx = 0

    # Best-path churn: does the argmax-bandwidth path change from hour to hour?
    prev_best: Dict[Tuple[int, int], int] = {}
    churn_events = 0
    churn_chances = 0
    rows: List[Dict[str, Any]] = []

    for hour_idx in hours:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            n = len(env.available_paths)
            if n == 0:
                continue
            pms = env.path_metrics_snapshot()
            bws = np.array([float(m.get("bandwidth_mbps") or 0.0) for m in pms])
            lats = np.array([float(m.get("latency_ms", 50.0)) for m in pms])
            losses = np.array([float(m.get("loss_rate", 0.0)) for m in pms])
            n_paths_list.append(n)
            n_ctx += 1

            if bws.max() <= 0.001:
                dead_contexts += 1
                continue

            best_bw = float(bws.max())
            med_bw = float(np.median(bws))
            bw_best.append(best_bw)
            bw_spread_abs.append(best_bw - float(bws.min()))
            bw_ratio_best_median.append(best_bw / med_bw if med_bw > 1e-9 else np.inf)
            bw_ratio_best_worst.append(
                best_bw / float(bws.min()) if bws.min() > 1e-9 else np.inf
            )
            bw_cv.append(float(bws.std() / bws.mean()) if bws.mean() > 1e-9 else 0.0)
            lat_spread_abs.append(float(lats.max() - lats.min()))
            lat_ratio_worst_best.append(
                float(lats.max() / lats.min()) if lats.min() > 1e-9 else np.inf
            )

            i_bw = int(np.argmax(bws))
            i_lat = int(np.argmin(lats))
            i_loss = int(np.argmin(losses))
            same_bw_lat += int(i_bw == i_lat)
            same_bw_loss += int(i_bw == i_loss)
            same_lat_loss += int(i_lat == i_loss)
            all_three_same += int(i_bw == i_lat == i_loss)

            key = (int(pair[0]), int(pair[1]))
            if key in prev_best:
                churn_chances += 1
                churn_events += int(prev_best[key] != i_bw)
            prev_best[key] = i_bw

            if len(rows) < 2000:
                rows.append(
                    {
                        "src": int(pair[0]),
                        "dst": int(pair[1]),
                        "hour_idx": int(hour_idx),
                        "n_paths": n,
                        "bw_best_mbps": best_bw,
                        "bw_median_mbps": med_bw,
                        "bw_min_mbps": float(bws.min()),
                        "lat_best_ms": float(lats.min()),
                        "lat_worst_ms": float(lats.max()),
                        "argmax_bw": i_bw,
                        "argmin_lat": i_lat,
                        "argmin_loss": i_loss,
                    }
                )
        if n_ctx and n_ctx % (24 * max(1, len(eval_pairs))) == 0:
            progress(f"  spread: hour {hour_idx} ({n_ctx} contexts)")

    finite_ratio = [r for r in bw_ratio_best_median if np.isfinite(r)]
    n_live = len(bw_best)
    report: Dict[str, Any] = {
        "run_dir": str(run_path),
        "n_pairs": len(eval_pairs),
        "n_hours": len(hours),
        "n_contexts": n_ctx,
        "n_contexts_with_bandwidth": n_live,
        "frac_contexts_all_paths_dead": (
            (dead_contexts / n_ctx) if n_ctx else float("nan")
        ),
        "candidate_paths": {
            "mean": float(np.mean(n_paths_list)) if n_paths_list else float("nan"),
            "min": int(np.min(n_paths_list)) if n_paths_list else 0,
            "max": int(np.max(n_paths_list)) if n_paths_list else 0,
        },
        "goodput_spread_within_context": {
            "best_mbps_mean": float(np.mean(bw_best)) if n_live else float("nan"),
            "best_minus_worst_mbps_mean": (
                float(np.mean(bw_spread_abs)) if n_live else float("nan")
            ),
            "best_minus_worst_mbps_p50": _pct(bw_spread_abs, 50),
            "best_over_median_p10": _pct(finite_ratio, 10),
            "best_over_median_p50": _pct(finite_ratio, 50),
            "best_over_median_p90": _pct(finite_ratio, 90),
            "coefficient_of_variation_p50": _pct(bw_cv, 50),
            "coefficient_of_variation_mean": (
                float(np.mean(bw_cv)) if n_live else float("nan")
            ),
        },
        "latency_spread_within_context": {
            "worst_minus_best_ms_p50": _pct(lat_spread_abs, 50),
            "worst_minus_best_ms_mean": (
                float(np.mean(lat_spread_abs)) if n_live else float("nan")
            ),
            "worst_over_best_p50": _pct(
                [r for r in lat_ratio_worst_best if np.isfinite(r)], 50
            ),
        },
        "best_path_churn": {
            "comparisons": churn_chances,
            "changes": churn_events,
            "fraction_of_hours_best_bw_path_changes": (
                churn_events / churn_chances if churn_chances else float("nan")
            ),
        },
        "intent_optima_coincidence": {
            "frac_argmax_bw_equals_argmin_latency": (
                same_bw_lat / n_live if n_live else float("nan")
            ),
            "frac_argmax_bw_equals_argmin_loss": (
                same_bw_loss / n_live if n_live else float("nan")
            ),
            "frac_argmin_latency_equals_argmin_loss": (
                same_lat_loss / n_live if n_live else float("nan")
            ),
            "frac_all_three_identical": (
                all_three_same / n_live if n_live else float("nan")
            ),
        },
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument("--hour-stride", type=int, default=1)
    parser.add_argument("--pairs-json", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "gap" / "realism")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = None
    if args.pairs_json:
        with open(args.pairs_json) as f:
            pairs = [(int(a), int(b)) for a, b in json.load(f)["pairs"]]

    print(f"Candidate-path spread within decision contexts on {run_path}")
    report, rows = analyze(
        run_path,
        max_pairs=args.max_pairs,
        hour_stride=args.hour_stride,
        pairs=pairs,
    )

    json_path = out_dir / "pair_selection_spread.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    if rows:
        import csv as _csv

        csv_path = out_dir / "pair_selection_spread_sample.csv"
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  saved: {csv_path}")

    g = report["goodput_spread_within_context"]
    c = report["intent_optima_coincidence"]
    print(f"  contexts                     : {report['n_contexts']}")
    print(
        f"  best-vs-median goodput ratio : p10 {g['best_over_median_p10']:.2f}  "
        f"p50 {g['best_over_median_p50']:.2f}  p90 {g['best_over_median_p90']:.2f}"
    )
    print(
        f"  best-minus-worst goodput     : {g['best_minus_worst_mbps_mean']:.0f} Mbps mean"
    )
    print(
        f"  best-bw path changes hourly  : "
        f"{report['best_path_churn']['fraction_of_hours_best_bw_path_changes']:.1%}"
    )
    print(
        f"  argmax-bw == argmin-latency  : {c['frac_argmax_bw_equals_argmin_latency']:.1%}"
    )
    print(
        f"  argmax-bw == argmin-loss     : {c['frac_argmax_bw_equals_argmin_loss']:.1%}"
    )
    print(f"  saved: {json_path}")


if __name__ == "__main__":
    main()
