"""Smoke tests for the enhanced DQN agent.

Verifies the agent can be instantiated, act on a zero state, store
experience, and exposes the networks used by ``04_train_dqn.py``/
``05_evaluate_methods.py`` checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig


@pytest.fixture
def agent():
    config = EnhancedDQNConfig(
        hidden_dim=16,
        n_hidden_layers=2,
        batch_size=4,
        buffer_size=64,
        min_buffer_size=4,
        use_batch_norm=False,
        use_prioritized_replay=False,
    )
    return EnhancedDQNAgent(state_dim=5, action_dim=3, config=config)


def test_act_returns_valid_action(agent):
    state = np.zeros(5, dtype=np.float32)
    action = agent.act(state)
    assert 0 <= int(action) < 3


def test_agent_exposes_expected_modules(agent):
    assert hasattr(agent, "q_network")
    assert hasattr(agent, "target_network")
    assert hasattr(agent, "optimizer")
    assert hasattr(agent, "scheduler")


def test_agent_remember_is_no_error(agent):
    state = np.zeros(5, dtype=np.float32)
    next_state = np.ones(5, dtype=np.float32)
    agent.remember(state, 0, 1.0, next_state, False)
