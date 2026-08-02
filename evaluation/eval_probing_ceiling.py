#!/usr/bin/env python3
"""Probing overhead and single-path performance ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.intent_cond_eval import (
    CONDITIONAL_CHECKPOINTS,
    DEFAULT_CONDITIONAL_AGENT,
    MAX_EVAL_PAIRS,
    run_probing_ceiling,
)
from src.pipeline.run_dirs import resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact output dir (default: <run>/intent_cond_latest).",
    )
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument("--congestion-bins", type=int, default=6)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Intents to compare under (default: balanced_extreme). The first "
        "drives probing_quality.csv / ceiling_by_congestion.csv.",
    )
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=None,
        help="JSON file with a 'pairs' list of [src, dst] to evaluate on.",
    )
    parser.add_argument(
        "--agent",
        choices=sorted(CONDITIONAL_CHECKPOINTS),
        default=DEFAULT_CONDITIONAL_AGENT,
        help="Conditional agent to profile as the learned selector.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "intent_cond_latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = None
    if args.pairs_json:
        with open(args.pairs_json) as f:
            pairs = [(int(a), int(b)) for a, b in json.load(f)["pairs"]]

    print(f"Probing overhead + single-path ceiling on {run_path}")
    result = run_probing_ceiling(
        run_path,
        out_dir,
        max_pairs=args.max_pairs,
        n_congestion_bins=args.congestion_bins,
        progress=print,
        profiles=args.profiles,
        pairs=pairs,
        agent_key=args.agent,
    )
    print("  quality vs probe cost:")
    for row in result["quality_rows"]:
        print(
            f"    {row['method_label']:<18} probe {row['probe_cost_per_selection_ms']:8.1f} ms/sel  "
            f"({row['probes_per_selection']:5.1f} probes)  "
            f"goodput {row['goodput_mean_mbps']:8.1f} Mbps"
        )
    print(f"  saved: {result['quality_csv']}")
    print(f"  saved: {result['ceiling_csv']}")
    if result.get("quality_by_intent_csv"):
        print(f"  saved: {result['quality_by_intent_csv']}")


if __name__ == "__main__":
    main()
