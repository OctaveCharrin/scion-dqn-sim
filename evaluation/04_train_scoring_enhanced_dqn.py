#!/usr/bin/env python3
"""Train the enhanced path-scoring DQN (Dueling + PER + Double DQN)."""

from src.pipeline.dqn_train_cli import run_scoring_training

if __name__ == "__main__":
    run_scoring_training("enhanced", "Enhanced path-scoring DQN")
