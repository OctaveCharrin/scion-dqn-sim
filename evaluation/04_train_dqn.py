#!/usr/bin/env python3
"""Train the enhanced flat DQN (multi-pair, action masking)."""

from src.pipeline.dqn_train_cli import run_flat_training

if __name__ == "__main__":
    run_flat_training("enhanced", "Enhanced DQN")
