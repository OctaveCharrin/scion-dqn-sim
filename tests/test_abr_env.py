"""Tests for the single-path ABR env."""

from __future__ import annotations

import numpy as np

from src.ns3env.abr import BITRATE_LADDER_MBPS
from src.ns3env.abr_env import (
    C_IS_LAST,
    C_QUALITY,
    GLOBAL_DIM,
    PATH_FEATURE_DIM,
    VideoAbrEnv,
)
from src.ns3env.dataplane import MockTraceConfig, MockTraceDataPlane


def _env(**cfg):
    cfg.setdefault("num_paths", 1)
    cfg.setdefault("episode_segments", 10)
    dp = MockTraceDataPlane(MockTraceConfig(**cfg))
    return VideoAbrEnv(dp)


def test_observation_contract():
    env = _env()
    obs = env.reset(seed=1)
    assert obs["global"].shape == (GLOBAL_DIM,)
    assert obs["paths"].shape == (len(BITRATE_LADDER_MBPS), PATH_FEATURE_DIM)
    assert obs["paths"].dtype == np.float32
    assert env.num_actions == len(BITRATE_LADDER_MBPS)


def test_quality_column_is_increasing_in_bitrate():
    env = _env()
    obs = env.reset(seed=1)
    q = obs["paths"][:, C_QUALITY]
    assert np.all(np.diff(q) > 0)  # VMAF rises with rung


def test_step_reports_vmaf_and_buffer_and_done():
    env = _env(episode_segments=5)
    env.reset(seed=2)
    done = False
    steps = 0
    while not done:
        obs, reward, done, info = env.step(3)
        assert -1.0 <= reward <= 1.0
        assert info["chosen_bitrate_mbps"] == BITRATE_LADDER_MBPS[3]
        assert info["chosen_vmaf"] > 0.0
        assert info["rebuffer_s"] >= 0.0
        assert info["buffer_s"] >= 0.0
        steps += 1
        assert steps <= 50
    assert steps == 5


def test_last_chosen_flag_tracks_action():
    env = _env()
    env.reset(seed=3)
    obs, _, _, _ = env.step(4)
    assert obs["paths"][4, C_IS_LAST] == 1.0
    assert obs["paths"][0, C_IS_LAST] == 0.0


def test_highest_bitrate_can_cause_rebuffering():
    # Force a slow path so the top rung overruns the buffer at least once.
    env = _env(num_paths=1, base_mbps=(0.5,), amp=(0.1,))
    env.reset(seed=4)
    total_rebuf = 0.0
    for _ in range(8):
        _, _, done, info = env.step(len(BITRATE_LADDER_MBPS) - 1)
        total_rebuf += info["rebuffer_s"]
        if done:
            break
    assert total_rebuf > 0.0


def test_invalid_action_raises():
    env = _env()
    env.reset(seed=5)
    try:
        env.step(len(BITRATE_LADDER_MBPS))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range action")
