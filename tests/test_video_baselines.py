"""Tests for the multipath baseline selectors."""

from __future__ import annotations

import numpy as np

from src.baselines.multipath_baselines import (
    MULTIPATH_SELECTORS,
    MaxThroughputSelector,
    MinRTTSelector,
    RoundRobinSelector,
    SingleBestPathSelector,
    StaticPathSelector,
)
from src.ns3env.video_env import GLOBAL_DIM


def _obs(paths):
    return {
        "global": np.zeros(GLOBAL_DIM, dtype=np.float32),
        "paths": np.asarray(paths, dtype=np.float32),
    }


def test_round_robin_cycles():
    sel = RoundRobinSelector()
    obs = _obs([[0, 0, 0, 0, 0]] * 3)
    assert [sel.select(obs) for _ in range(5)] == [0, 1, 2, 0, 1]


def test_min_rtt_picks_lowest_rtt():
    # cols: throughput_norm, rtt_norm, loss, throughput_ratio, last_chosen
    obs = _obs([[0.5, 0.9, 0, 0.5, 0], [0.5, 0.2, 0, 0.5, 0], [0.5, 0.5, 0, 0.5, 0]])
    assert MinRTTSelector().select(obs) == 1


def test_max_throughput_picks_highest():
    obs = _obs([[0.3, 0.1, 0, 0.3, 0], [0.9, 0.1, 0, 0.9, 0], [0.5, 0.1, 0, 0.5, 0]])
    assert MaxThroughputSelector().select(obs) == 1


def test_static_path_clamps():
    sel = StaticPathSelector(path_idx=5)
    obs = _obs([[0.5, 0.1, 0, 0.5, 0]] * 2)
    assert sel.select(obs) == 1  # clamped to n-1


def test_single_best_locks():
    sel = SingleBestPathSelector()
    obs1 = _obs([[0.2, 0, 0, 0.2, 0], [0.9, 0, 0, 0.9, 0]])
    assert sel.select(obs1) == 1
    # Even if path 0 later looks better, it stays locked.
    obs2 = _obs([[0.99, 0, 0, 0.99, 0], [0.1, 0, 0, 0.1, 0]])
    assert sel.select(obs2) == 1


def test_registry_complete():
    for name, cls in MULTIPATH_SELECTORS.items():
        sel = cls()
        assert hasattr(sel, "select")
