"""Tests for compact link traffic state storage."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.simulation.link_state_store import (
    LINK_STATE_FORMAT,
    LinkTrafficState,
    load_link_traffic_state,
    pack_hourly_file,
    save_link_traffic_state,
)
from src.simulation.traffic_metrics import LinkMetricArrays


def test_compact_format_small_on_disk(tmp_path: Path):
    link_keys = [(0, 1), (1, 2)]
    hours = {
        0: {
            "latency_ms": np.array([1.0, 2.0]),
            "available_mbps": np.array([100.0, 200.0]),
            "utilization": np.array([0.1, 0.2]),
            "utilization_raw": np.array([0.1, 0.2]),
            "loss_rate": np.array([0.0, 0.0]),
        }
    }
    state = LinkTrafficState(link_keys=link_keys, hours=hours)
    path = tmp_path / "link_states.pkl"
    save_link_traffic_state(path, state)
    assert path.stat().st_size < 500_000
    loaded = load_link_traffic_state(path)
    assert not loaded.is_legacy
    block = loaded.path_metrics_block(0, 0, 2, {(0, 2): [[0], [1]]})
    assert "path_0" in block


def test_legacy_format_still_loads():
    legacy = {
        0: {
            "by_pair": {
                "pair_1_3": {
                    "path_0": {
                        "latency_ms": 10.0,
                        "available_bandwidth_mbps": 500.0,
                        "loss_rate": 0.0,
                        "utilization": 0.1,
                    }
                }
            }
        }
    }
    state = LinkTrafficState(link_keys=[], hours={}, legacy_by_hour=legacy)
    block = state.path_metrics_block(0, 1, 3, {})
    assert block["path_0"]["latency_ms"] == pytest.approx(10.0)
