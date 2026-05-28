#!/usr/bin/env python3
"""Train the enhanced flat DQN (multi-pair, action masking)."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from _common import resolve_run_dir
from train_lib import train_flat_dqn


def main() -> None:
    run_path = Path(resolve_run_dir())
    print(f"\nTraining Enhanced DQN on {run_path}")
    stats = train_flat_dqn(run_path, "enhanced")
    print(f"  avg train reward: {stats['avg_reward']:.3f}")
    print(f"  saved: {stats['checkpoint']}")


if __name__ == "__main__":
    main()
