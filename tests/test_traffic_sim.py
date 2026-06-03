"""Tests for calibrated traffic simulation."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.simulation.traffic_config import TrafficSimConfig
from src.simulation.traffic_metrics import (
    compute_link_metrics_vectorized,
    summarize_link_loads,
)


def test_scaled_base_rate_scales_with_pool():
    cfg = TrafficSimConfig(base_rate_mbps=100.0, reference_pair_pool_size=32)
    small = cfg.scaled_base_rate_mbps(32)
    large = cfg.scaled_base_rate_mbps(320)
    assert small == pytest.approx(100.0)
    assert large == pytest.approx(10.0)


def test_background_pairs_bounded():
    cfg = TrafficSimConfig(background_min=20, background_max=50)
    assert cfg.background_pairs_per_hour(50, 100) <= 50
    assert cfg.background_pairs_per_hour(10, 80) >= 20


def test_link_metrics_util_raw_can_exceed_cap():
    loads = np.array([2000.0])
    caps = np.array([1000.0])
    lats = np.array([10.0])
    m = compute_link_metrics_vectorized(loads, caps, lats, util_cap=1.5)
    assert m.utilization_capped[0] == pytest.approx(1.5)
    assert m.utilization_raw[0] == pytest.approx(2.0)


def test_summarize_link_loads_reports_raw_util():
    caps = np.array([100.0, 100.0])
    loads_by_hour = {
        0: np.array([50.0, 0.0]),
        1: np.array([150.0, 10.0]),
    }
    s = summarize_link_loads(loads_by_hour, caps, peak_hour=1)
    assert s["util_raw_max"] == pytest.approx(1.5)
    assert s["fraction_links_util_gt_1"] > 0


def test_simulate_link_traffic_mini_run(tmp_path: Path):
    """End-to-end smoke on a tiny topology."""
    from src.simulation.beacon_pipeline import run_beaconing
    from src.simulation.link_traffic_sim import simulate_link_traffic

    G = nx.Graph()
    for nid, isd, role in [(0, 0, "core"), (1, 0, "non-core"), (2, 0, "non-core")]:
        G.add_node(nid, isd=isd, role=role, x=float(nid), y=0.0)
    G.add_edge(0, 1, type="parent_child", latency=10.0, bandwidth=1000.0)
    G.add_edge(1, 2, type="parent_child", latency=8.0, bandwidth=1000.0)

    topo_dir = tmp_path / "topology"
    topo_dir.mkdir()
    with open(topo_dir / "scion_topology.json", "w") as f:
        json.dump(
            {"isds": [0], "core_ases": [0], "graph": nx.node_link_data(G)},
            f,
        )

    run_beaconing(tmp_path)

    cfg = TrafficSimConfig(
        num_days=2,
        active_pairs_min=1,
        active_pairs_max=2,
        background_max=5,
        write_link_states_json=False,
    )
    meta = simulate_link_traffic(tmp_path, cfg)
    assert (tmp_path / "link_states.pkl").is_file()
    assert meta["traffic_calibration"]["scaled_base_rate_mbps"] > 0
    assert "link_utilization" in meta
