#!/usr/bin/env python3
"""Train reward-weight-conditioned path-scoring DQN (multi-objective)."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.rl.path_selection_train import ScoringHyperparams, train_conditional_scoring_dqn


def main() -> None:
    run_path = Path(resolve_run_dir())
    print(f"\nTraining conditional path-scoring DQN on {run_path}")
    hp = ScoringHyperparams.from_env()
    stats = train_conditional_scoring_dqn(run_path, hp)
    print(f"  avg train reward: {stats['avg_reward']:.3f}")
    print(f"  profiles: {', '.join(stats['training_profiles'])}")
    print(f"  saved: {stats['checkpoint']}")


if __name__ == "__main__":
    main()
