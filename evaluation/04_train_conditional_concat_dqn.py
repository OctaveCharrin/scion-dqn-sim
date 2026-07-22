#!/usr/bin/env python3
"""Train the reward-weight-conditioned path-scoring DQN with plain dueling-concat.

This is the ablation counterpart to ``04_train_conditional_dqn.py`` (FiLM): it
conditions on the reward-weight (intent) vector by concatenating it into the global
state and feeding the base dueling network, rather than FiLM-modulating per-path
features. It writes ``dqn_conditional_concat_model.pth`` (arch tag ``dueling_concat``).
"""

from src.pipeline.dqn_train_cli import run_conditional_training

if __name__ == "__main__":
    run_conditional_training(architecture="concat")
