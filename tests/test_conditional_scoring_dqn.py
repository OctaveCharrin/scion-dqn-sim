"""Tests for reward-weight-conditioned path-scoring DQN."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import ConditionalPathScoringDQNAgent
from src.rl.reward_profiles import REWARD_PROFILES, get_profile, sample_training_weights
from src.rl.reward_profiles import (
    DISTINCTIVE_REWARD_PROFILES,
    get_conditional_training_profiles,
    stratified_training_profile_schedule,
)
from src.simulation.evaluation_env import (
    CONDITIONAL_SCORING_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    REWARD_WEIGHT_DIM,
    SCORING_GLOBAL_DIM,
    encode_reward_weights,
    encode_reward_weights_for_conditional,
    encode_reward_weights_policy,
    set_conditional_weight_encoding,
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
    set_conditional_weight_encoding("policy")
    base = np.random.randn(SCORING_GLOBAL_DIM).astype(np.float32)
    wvec = encode_reward_weights_for_conditional(get_profile("latency").weights)
    state = {
        "global": np.concatenate([base, wvec]),
        "paths": np.random.randn(4, PATH_FEATURE_DIM).astype(np.float32),
    }
    a = agent.act(state, evaluate=True)
    assert 0 <= a < 4


def test_policy_weight_encoding_differs_from_raw() -> None:
    w = RewardWeights(w1=0.95, w2=0.05, w3=0.85, w4=0.15, w_probe=0.25)
    raw = encode_reward_weights(w)
    pol = encode_reward_weights_policy(w)
    assert not np.allclose(raw, pol)


def test_distinctive_profiles_more_separated() -> None:
    bw = next(p for p in DISTINCTIVE_REWARD_PROFILES if p.name == "bandwidth_max")
    loss = next(p for p in DISTINCTIVE_REWARD_PROFILES if p.name == "loss_averse")
    assert bw.weights.w1 - loss.weights.w1 > 0.5


def test_stratified_schedule_covers_all_profiles() -> None:
    import random

    profiles = get_conditional_training_profiles()
    sched = stratified_training_profile_schedule(60, profiles, random.Random(0))
    assert len(sched) == 60
    assert {p.name for p in sched} == {p.name for p in profiles}


def test_weight_film_changes_q_values(agent: ConditionalPathScoringDQNAgent) -> None:
    """Different weights must change per-path Q (not only a shared scalar offset)."""
    import torch

    set_conditional_weight_encoding("policy")
    scoring = np.zeros(SCORING_GLOBAL_DIM, dtype=np.float32)
    paths = np.array(
        [
            [0.2, 0.01, 0.1, 0.95, 0.1, 0.9, 0.9],
            [0.15, 0.02, 0.1, 0.4, 0.2, 0.5, 0.5],
            [0.9, 0.35, 0.15, 0.35, 0.5, 0.4, 0.2],
        ],
        dtype=np.float32,
    )
    bw_w = encode_reward_weights_for_conditional(
        RewardWeights(w1=1.0, w2=0.0, w3=0.05, w4=0.05, w_probe=0.05)
    )
    lat_w = encode_reward_weights_for_conditional(
        RewardWeights(w1=0.1, w2=0.9, w3=0.15, w4=1.0, w_probe=0.05)
    )
    g_bw = torch.as_tensor(
        np.concatenate([scoring, bw_w])[None, :],
        dtype=torch.float32,
        device=agent.device,
    )
    g_lat = torch.as_tensor(
        np.concatenate([scoring, lat_w])[None, :],
        dtype=torch.float32,
        device=agent.device,
    )
    pf = torch.as_tensor(paths, dtype=torch.float32, device=agent.device).unsqueeze(0)
    mask = torch.ones(1, paths.shape[0], dtype=torch.bool, device=agent.device)
    with torch.no_grad():
        q_bw = agent.q_network(g_bw, pf, mask)
        q_lat = agent.q_network(g_lat, pf, mask)
    assert not torch.allclose(q_bw, q_lat)
