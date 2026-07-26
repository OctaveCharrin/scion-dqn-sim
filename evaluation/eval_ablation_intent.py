#!/usr/bin/env python3
"""Intent-conditioning ablation across the agent ladder (Chapter 6, Table 6.1):
Flat DQN, unconditioned path-scoring DQN, Value-Concat, Two-Stream-Concat, FiLM,
and a per-context reward oracle.

Writes ``ablation_reward_matrix.csv``, ``ablation_behavioral_divergence.csv``,
``ablation_per_context_rewards.npz`` and ``table_6_1.tex`` into an output
directory (default: ``<run>/chapter6_latest``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.chapter6_eval import INTENT_PROFILES, MAX_EVAL_PAIRS, run_ablation
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
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help=f"Reward profiles to evaluate as intents (default: {' '.join(INTENT_PROFILES)}).",
    )
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=None,
        help="JSON file with a 'pairs' list of [src, dst] to evaluate on "
        "(e.g. a genuinely held-out set from build_heldout_pairs.py).",
    )
    parser.add_argument(
        "--hour-stride",
        type=int,
        default=1,
        help="Evaluate every Nth held-out hour (default 1 = all 336).",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip the per-context reward oracle row.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "chapter6_latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = None
    if args.pairs_json:
        with open(args.pairs_json) as f:
            pairs = [(int(a), int(b)) for a, b in json.load(f)["pairs"]]
        print(f"  using {len(pairs)} pairs from {args.pairs_json}")

    print(f"Ablation (intent conditioning) on {run_path}")
    result = run_ablation(
        run_path,
        out_dir,
        max_pairs=args.max_pairs,
        progress=print,
        profiles=args.profiles,
        pairs=pairs,
        hour_stride=args.hour_stride,
        include_oracle=not args.no_oracle,
    )
    print(f"  methods: {', '.join(result['methods'])}")
    print("  behavioral divergence:")
    for row in result["divergence_rows"]:
        print(f"    {row['method_label']:<22} {row['behavioral_divergence']:.3f}")
    print(f"  saved: {result['reward_csv']}")
    print(f"  saved: {result['divergence_csv']}")
    print(f"  saved: {result['table_tex']}")


if __name__ == "__main__":
    main()
