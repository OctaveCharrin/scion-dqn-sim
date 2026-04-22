"""Tests for the path services (aggregation helpers)."""

from __future__ import annotations

import numpy as np

from src.path_services.pathprobe import PathProbe


def test_aggregate_latency_is_sum():
    assert PathProbe._aggregate_latency(np.array([5.0, 3.0, 7.0, 2.0])) == 17.0


def test_aggregate_bandwidth_is_min():
    assert PathProbe._aggregate_bandwidth(np.array([1000.0, 500.0, 750.0])) == 500.0


def test_aggregate_bandwidth_empty_returns_zero():
    assert PathProbe._aggregate_bandwidth(np.array([])) == 0.0


def test_aggregate_loss_is_compound():
    losses = np.array([0.01, 0.02, 0.0, 0.03])
    expected = 1 - (0.99 * 0.98 * 1.0 * 0.97)
    assert abs(PathProbe._aggregate_loss(losses) - expected) < 1e-6


def test_aggregate_loss_zero_when_all_links_clean():
    assert PathProbe._aggregate_loss(np.zeros(5)) == 0.0
