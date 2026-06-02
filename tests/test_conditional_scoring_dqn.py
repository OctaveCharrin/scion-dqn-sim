"""Tests for reward-weight-conditioned path-scoring DQN."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import ConditionalPathScoringDQNAgent
from src.rl.reward_profiles import REWARD_PROFILES, get_profile, sample_training_weights
from src.simulation.evaluation_env import (
    CONDITIONAL_SCORING_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    REWARD_WEIGHT_DIM,
    SCORING_GLOBAL_DIM,
    encode_reward_weights,
    RewardWeights,
)


@pytest.fixture
def agent() -> ConditionalPathScoringDQNAgent:
    cfg = EnhancedDQNConfig(
        hidden_dim=32,
        n_hidden_layers=2,
        batch_size=4,
        buffer_size=64,
        min_buffer_size=4,
        use_batch_norm=False,
        use_prioritized_replay=False,
    )
    return ConditionalPathScoringDQNAgent(
        global_dim=CONDITIONAL_SCORING_GLOBAL_DIM,
        path_dim=PATH_FEATURE_DIM,
        config=cfg,
    )


def test_global_dim_includes_weights() -> None:
    assert CONDITIONAL_SCORING_GLOBAL_DIM == SCORING_GLOBAL_DIM + REWARD_WEIGHT_DIM


def test_encode_reward_weights() -> None:
    w = RewardWeights(w1=0.9, w2=0.1, w3=0.4, w4=0.6, w_probe=0.02)
    vec = encode_reward_weights(w)
    assert vec.shape == (REWARD_WEIGHT_DIM,)
    assert vec.dtype == np.float32
    assert float(vec[0]) == pytest.approx(0.9)


def test_reward_profiles_unique_names() -> None:
    names = [p.name for p in REWARD_PROFILES]
    assert len(names) == len(set(names))
    assert get_profile("throughput").weights.w1 > get_profile("trust_quality").weights.w1


def test_sample_training_weights() -> None:
    import random

    rng = random.Random(0)
    seen = {sample_training_weights(rng).w1 for _ in range(50)}
    assert len(seen) > 1


def test_conditional_agent_act(agent: ConditionalPathScoringDQNAgent) -> None:
    base = np.random.randn(SCORING_GLOBAL_DIM).astype(np.float32)
    wvec = encode_reward_weights(get_profile("latency").weights)
    state = {
        "global": np.concatenate([base, wvec]),
        "paths": np.random.randn(4, PATH_FEATURE_DIM).astype(np.float32),
    }
    a = agent.act(state, evaluate=True)
    assert 0 <= a < 4
