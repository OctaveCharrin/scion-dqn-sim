#!/usr/bin/env python3
"""Train reward-weight-conditioned path-scoring DQN (multi-objective)."""

from src.pipeline.dqn_train_cli import run_conditional_training

if __name__ == "__main__":
    run_conditional_training()
