#!/usr/bin/env python3
"""Train the naive value-only-concat conditional path-scoring DQN (ablation).

This is the failure-mode baseline for the Chapter 6 conditioning study. The
reward-weight (intent) vector is concatenated into the *value* stream only; the
per-path advantage stream never sees the intent. Since the selected path is
``argmax_i A(s, path_i)`` and the intent can only shift ``V(s)`` uniformly, this
network is structurally unable to re-rank paths as the intent changes -- the
contrast against FiLM (``04_train_conditional_dqn.py``). It writes
``dqn_conditional_value_concat_model.pth`` (arch tag ``value_only_concat``).
"""

from src.pipeline.dqn_train_cli import run_conditional_training

if __name__ == "__main__":
    run_conditional_training(architecture="value_concat")
