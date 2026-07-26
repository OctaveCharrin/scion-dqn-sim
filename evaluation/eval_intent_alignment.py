#!/usr/bin/env python3
"""Intent alignment for one conditional checkpoint (Chapter 6, Figure 6.1).

Writes ``intent_reward_matrix.csv`` (the 5x5 R(intent_told, intent_scored) matrix)
and ``intent_selection_metrics.csv`` (chosen-path latency/bandwidth/trust per
conditioning intent) into an output directory (default: ``<run>/chapter6_latest``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.chapter6_eval import (
    CONDITIONAL_CHECKPOINTS,
    DEFAULT_CONDITIONAL_AGENT,
    MAX_EVAL_PAIRS,
    run_intent_alignment,
)
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
        "--agent",
        choices=sorted(CONDITIONAL_CHECKPOINTS),
        default=DEFAULT_CONDITIONAL_AGENT,
        help="Conditional agent to profile.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "chapter6_latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Intent alignment ({args.agent}) on {run_path}")
    result = run_intent_alignment(
        run_path,
        out_dir,
        max_pairs=args.max_pairs,
        progress=print,
        agent_key=args.agent,
    )
    print(f"  saved: {result['matrix_csv']}")
    print(f"  saved: {result['metrics_csv']}")


if __name__ == "__main__":
    main()
