#!/usr/bin/env python3
"""Ablation: Flat DQN vs Conditional-Concat vs Conditional-FiLM across the five
named reward profiles (Chapter 6, Table 6.1).

Writes ``ablation_reward_matrix.csv``, ``ablation_behavioral_divergence.csv`` and
``table_6_1.tex`` into an output directory (default: ``<run>/chapter6_latest``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.chapter6_eval import MAX_EVAL_PAIRS, run_ablation
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
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "chapter6_latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ablation (intent conditioning) on {run_path}")
    result = run_ablation(run_path, out_dir, max_pairs=args.max_pairs, progress=print)
    print(f"  methods: {', '.join(result['methods'])}")
    print("  behavioral divergence:")
    for row in result["divergence_rows"]:
        print(f"    {row['method_label']:<22} {row['behavioral_divergence']:.3f}")
    print(f"  saved: {result['reward_csv']}")
    print(f"  saved: {result['divergence_csv']}")
    print(f"  saved: {result['table_tex']}")


if __name__ == "__main__":
    main()
