#!/usr/bin/env python3
"""Run the full evaluation pipeline (steps 01–06) in order."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from src.pipeline.run_dirs import (
    PIPELINE_STEP_SCRIPTS,
    pipeline_steps_from,
    resolve_existing_run_dir,
    run_script,
)

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SCION DQN evaluation pipeline.")
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        metavar="N",
        help=f"Start at pipeline step N (1–{len(PIPELINE_STEP_SCRIPTS)}, default: 1).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing or target run directory (default: create run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=None,
        metavar="N",
        help="If set, export DQN_TRAIN_EPISODES=N for all 04_train_* steps.",
    )
    args = parser.parse_args()

    if args.train_episodes is not None:
        os.environ["DQN_TRAIN_EPISODES"] = str(args.train_episodes)

    if args.run_dir:
        run_dir = resolve_existing_run_dir(str(args.run_dir), cwd=EVAL_DIR)
    elif args.from_step > 1:
        run_dir = resolve_existing_run_dir(cwd=EVAL_DIR)
    else:
        run_dir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(EVAL_DIR / run_dir, exist_ok=True)

    print(f"Pipeline run directory: {run_dir}")
    print(f"Starting from step {args.from_step}")

    for script in pipeline_steps_from(args.from_step):
        run_script(script, run_dir, cwd=EVAL_DIR)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
