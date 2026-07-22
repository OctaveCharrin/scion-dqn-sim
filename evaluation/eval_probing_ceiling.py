#!/usr/bin/env python3
"""Probing overhead and single-path performance ceiling (Chapter 6, Figs 6.2/6.3).

Compares Conditional-FiLM against heuristics (Shortest-Path, Widest-Path,
Lowest-Latency, Random). Writes ``probing_quality.csv`` (quality vs probe cost)
and ``ceiling_by_congestion.csv`` (QoE vs realized congestion) into an output
directory (default: ``<run>/chapter6_latest``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.chapter6_eval import MAX_EVAL_PAIRS, run_probing_ceiling
from src.pipeline.run_dirs import resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact output dir (default: <run>/chapter6_latest).",
    )
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument("--congestion-bins", type=int, default=6)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "chapter6_latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probing overhead + single-path ceiling on {run_path}")
    result = run_probing_ceiling(
        run_path,
        out_dir,
        max_pairs=args.max_pairs,
        n_congestion_bins=args.congestion_bins,
        progress=print,
    )
    print("  quality vs probe cost:")
    for row in result["quality_rows"]:
        print(
            f"    {row['method_label']:<18} probe {row['probe_cost_per_selection_ms']:8.1f} ms/sel  "
            f"goodput {row['goodput_mean_mbps']:8.1f} Mbps"
        )
    print(f"  saved: {result['quality_csv']}")
    print(f"  saved: {result['ceiling_csv']}")


if __name__ == "__main__":
    main()
