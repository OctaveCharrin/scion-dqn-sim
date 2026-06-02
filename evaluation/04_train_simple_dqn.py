#!/usr/bin/env python3
"""Train the simple flat DQN."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.rl.path_selection_train import train_flat_dqn


def main() -> None:
    run_path = Path(resolve_run_dir())
    print(f"\nTraining Simple DQN on {run_path}")
    stats = train_flat_dqn(run_path, "simple")
    print(f"  avg train reward: {stats['avg_reward']:.3f}")
    print(f"  saved: {stats['checkpoint']}")


if __name__ == "__main__":
    main()
