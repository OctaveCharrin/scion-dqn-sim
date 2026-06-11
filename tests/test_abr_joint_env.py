"""Tests for the joint (path x bitrate) multipath ABR env."""

from __future__ import annotations

import numpy as np

from src.ns3env.abr import BITRATE_LADDER_MBPS
from src.ns3env.abr_joint_env import (
    CJ_IS_LAST_BR,
    CJ_IS_LAST_PATH,
    GLOBAL_DIM,
    PATH_FEATURE_DIM,
    VideoMultipathAbrEnv,
)
from src.ns3env.dataplane import MockTraceConfig, MockTraceDataPlane


def _env(num_paths=3, **cfg):
    cfg.setdefault("episode_segments", 10)
    dp = MockTraceDataPlane(MockTraceConfig(num_paths=num_paths, **cfg))
    return VideoMultipathAbrEnv(dp)


def test_joint_observation_contract():
    env = _env(num_paths=3)
    obs = env.reset(seed=1)
    b = len(BITRATE_LADDER_MBPS)
    assert env.num_actions == 3 * b
    assert obs["global"].shape == (GLOBAL_DIM,)
    assert obs["paths"].shape == (3 * b, PATH_FEATURE_DIM)


def test_action_decodes_to_path_and_bitrate():
    env = _env(num_paths=3)
    env.reset(seed=2)
    b = env.num_bitrates
    # action for (path=2, bitrate=4)
    action = 2 * b + 4
    obs, reward, done, info = env.step(action)
    assert info["chosen_path"] == 2
    assert info["bitrate_idx"] == 4
    assert info["chosen_bitrate_mbps"] == BITRATE_LADDER_MBPS[4]
    # last-path / last-bitrate flags reflect the choice.
    assert obs["paths"][2 * b + 4, CJ_IS_LAST_BR] == 1.0
    assert obs["paths"][2 * b + 0, CJ_IS_LAST_PATH] == 1.0


def test_joint_episode_runs_to_done():
    env = _env(num_paths=2, episode_segments=5)
    env.reset(seed=3)
    done = False
    steps = 0
    while not done:
        _, reward, done, info = env.step(0)
        assert -1.0 <= reward <= 1.0
        assert info["rebuffer_s"] >= 0.0
        steps += 1
        assert steps <= 50
    assert steps == 5


def test_invalid_action_raises():
    env = _env(num_paths=2)
    env.reset(seed=4)
    try:
        env.step(env.num_actions)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range action")
