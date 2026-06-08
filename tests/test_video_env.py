"""Tests for the multipath-QUIC video env and its mock data plane."""

from __future__ import annotations

import numpy as np
import pytest

from src.ns3env.dataplane import MockTraceConfig, MockTraceDataPlane
from src.ns3env.video_env import GLOBAL_DIM, PATH_FEATURE_DIM, Ns3VideoMpquicEnv


def make_env(**cfg):
    cfg.setdefault("episode_segments", 12)
    dp = MockTraceDataPlane(MockTraceConfig(**cfg))
    return Ns3VideoMpquicEnv(dp)


def test_observation_shapes():
    env = make_env(num_paths=3)
    obs = env.reset(seed=1)
    assert set(obs) == {"global", "paths"}
    assert obs["global"].shape == (GLOBAL_DIM,)
    assert obs["paths"].shape == (3, PATH_FEATURE_DIM)
    assert obs["global"].dtype == np.float32
    assert obs["paths"].dtype == np.float32


def test_step_contract_and_done():
    env = make_env(num_paths=2, episode_segments=5)
    env.reset(seed=2)
    done = False
    steps = 0
    last_obs = None
    while not done:
        obs, reward, done, info = env.step(steps % env.num_paths)
        last_obs = obs
        assert -1.0 <= reward <= 1.0
        assert info["chosen_path"] == steps % env.num_paths
        assert info["throughput_mbps"] > 0.0
        steps += 1
        assert steps <= 100  # guard against runaway
    assert steps == 5
    assert last_obs["paths"].shape == (2, PATH_FEATURE_DIM)


def test_features_are_normalized():
    env = make_env(num_paths=3)
    obs = env.reset(seed=3)
    for _ in range(8):
        obs, _, done, _ = env.step(0)
        paths = obs["paths"]
        assert np.all(paths[:, 0] >= 0.0) and np.all(
            paths[:, 0] <= 1.0
        )  # throughput_norm
        assert np.all(paths[:, 1] >= 0.0) and np.all(paths[:, 1] <= 1.0)  # rtt_norm
        assert np.all(paths[:, 2] >= 0.0) and np.all(paths[:, 2] <= 1.0)  # loss
        assert np.all(paths[:, 4] >= 0.0) and np.all(
            paths[:, 4] <= 1.0
        )  # last_chosen flag
        assert np.all((obs["global"] >= 0.0) & (obs["global"] <= 1.0))
        if done:
            break


def test_last_chosen_flag_tracks_action():
    env = make_env(num_paths=3)
    env.reset(seed=4)
    obs, _, _, _ = env.step(2)
    assert obs["paths"][2, 4] == 1.0
    assert obs["paths"][0, 4] == 0.0


def test_invalid_action_raises():
    env = make_env(num_paths=2)
    env.reset(seed=5)
    with pytest.raises(ValueError):
        env.step(2)


def test_determinism_same_seed():
    env_a = make_env(num_paths=3)
    env_b = make_env(num_paths=3)
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    for _ in range(10):
        oa, ra, da, _ = env_a.step(1)
        ob, rb, db, _ = env_b.step(1)
        assert ra == rb
        assert np.allclose(oa["paths"], ob["paths"])
        if da or db:
            assert da == db
            break


def test_clock_advances():
    env = make_env(num_paths=2)
    env.reset(seed=8)
    _, _, _, info0 = env.step(0)
    _, _, _, info1 = env.step(0)
    assert info1["clock_s"] > info0["clock_s"]
