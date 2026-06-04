#!/usr/bin/env python3
"""Train all DQN variants in one process (single ``link_states.pkl`` load)."""

from src.pipeline.dqn_train_cli import resolve_train_run_path, run_all_dqn_models

if __name__ == "__main__":
    import sys

    run_path = resolve_train_run_path(sys.argv[1] if len(sys.argv) > 1 else None)
    run_all_dqn_models(run_path)
