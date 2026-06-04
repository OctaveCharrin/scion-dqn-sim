#!/usr/bin/env python3
"""Train the simple path-scoring DQN."""

from src.pipeline.dqn_train_cli import run_scoring_training

if __name__ == "__main__":
    run_scoring_training("simple", "Simple path-scoring DQN")
