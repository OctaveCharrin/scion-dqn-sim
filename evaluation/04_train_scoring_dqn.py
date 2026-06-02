#!/usr/bin/env python3
"""Train the simple path-scoring DQN."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.rl.path_selection_train import ScoringHyperparams, train_scoring_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    hp = ScoringHyperparams.from_env()
    if args.config_json and args.config_json.is_file():
        with open(args.config_json) as f:
            hp = ScoringHyperparams.from_dict(json.load(f))

    num_eps = os.environ.get("DQN_TRAIN_EPISODES", "").strip()
    n_episodes = int(num_eps) if num_eps.isdigit() else None

    ckpt = args.checkpoint or run_path / "dqn_scoring_simple_model.pth"
    stats_path = run_path / "dqn_scoring_simple_training_stats.json"

    print(f"\nTraining Simple Path-Scoring DQN on {run_path}")
    stats = train_scoring_dqn(
        run_path,
        "simple",
        hp,
        num_episodes=n_episodes,
        checkpoint_path=ckpt,
        stats_path=stats_path,
    )
    print(f"  avg train reward: {stats['avg_reward']:.3f}")
    print(f"  saved: {ckpt}")


if __name__ == "__main__":
    main()
