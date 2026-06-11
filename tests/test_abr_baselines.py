"""Tests for the ABR heuristic selectors."""

from __future__ import annotations

import numpy as np

from src.baselines.abr_baselines import (
    ABR_SELECTORS,
    BufferBasedSelector,
    FixedBitrateSelector,
    RateBasedSelector,
)
from src.ns3env.abr_env import C_FEASIBLE, G_BUFFER, GLOBAL_DIM, PATH_FEATURE_DIM


def _obs(feasible, buffer_frac=0.0):
    paths = np.zeros((len(feasible), PATH_FEATURE_DIM), dtype=np.float32)
    paths[:, C_FEASIBLE] = feasible
    glob = np.zeros(GLOBAL_DIM, dtype=np.float32)
    glob[G_BUFFER] = buffer_frac
    return {"global": glob, "paths": paths}


def test_fixed_selectors_pick_bounds():
    obs = _obs([1, 1, 1, 1, 1, 1])
    assert FixedBitrateSelector(0.0).select(obs) == 0
    assert FixedBitrateSelector(1.0).select(obs) == 5
    assert FixedBitrateSelector(0.5).select(obs) in (2, 3)


def test_rate_based_picks_highest_sustainable():
    # Rungs 0..3 sustainable (feasible == 1), 4..5 not.
    obs = _obs([1, 1, 1, 1, 0.6, 0.3])
    assert RateBasedSelector().select(obs) == 3


def test_rate_based_falls_back_to_lowest():
    obs = _obs([0.5, 0.4, 0.2, 0.1, 0.05, 0.0])
    assert RateBasedSelector().select(obs) == 0


def test_buffer_based_low_buffer_picks_low_high_buffer_picks_high():
    low = _obs([1] * 6, buffer_frac=0.0)
    high = _obs([1] * 6, buffer_frac=1.0)
    assert BufferBasedSelector().select(low) == 0
    assert BufferBasedSelector().select(high) == 5


def test_registry_constructs_all():
    obs = _obs([1, 1, 1, 0, 0, 0], buffer_frac=0.5)
    for name, cls in ABR_SELECTORS.items():
        idx = cls().select(obs)
        assert 0 <= idx < 6, name
