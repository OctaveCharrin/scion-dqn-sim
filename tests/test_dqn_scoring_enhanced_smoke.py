"""Smoke tests for path-scoring Dueling Double DQN with PER."""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_enhanced import (
    DuelingPathScoringDQN,
    EnhancedPathScoringDQNAgent,
)

torch = pytest.importorskip("torch")

G, P = 5, 6


@pytest.fixture
def config() -> EnhancedDQNConfig:
    return EnhancedDQNConfig(
        hidden_dim=32,
        n_hidden_layers=2,
        batch_size=4,
        buffer_size=64,
        min_buffer_size=4,
        use_batch_norm=False,
        use_prioritized_replay=True,
        use_double_dqn=True,
    )


@pytest.fixture
def agent(config: EnhancedDQNConfig) -> EnhancedPathScoringDQNAgent:
    return EnhancedPathScoringDQNAgent(global_dim=G, path_dim=P, config=config)


def test_dueling_forward_masking(config: EnhancedDQNConfig) -> None:
    net = DuelingPathScoringDQN(G, P, config)
    g = torch.randn(2, G)
    paths = torch.randn(2, 5, P)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    q = net(g, paths, mask)
    assert q.shape == (2, 5)
    assert torch.all(q[~mask] <= -1e8)


def test_act_and_replay_variable_paths(agent: EnhancedPathScoringDQNAgent) -> None:
    for t in range(6):
        n, nn = 2 + t % 3, 3 + t % 4
        s = {
            "global": np.random.randn(G).astype(np.float32),
            "paths": np.random.randn(n, P).astype(np.float32),
        }
        ns = {
            "global": np.random.randn(G).astype(np.float32),
            "paths": np.random.randn(nn, P).astype(np.float32),
        }
        a = agent.act(s, evaluate=True)
        assert 0 <= a < n
        agent.remember(s, a, 1.0, ns, 0.0)

    loss = agent.replay()
    assert loss is not None
