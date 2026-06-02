"""Tests for evaluation beacon pipeline (step 02)."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from src.beacon.beacon_sim import BeaconSimulator
from src.simulation.beacon_pipeline import (
    discover_paths_for_topology,
    load_topology_graph,
    run_beaconing,
)


def _mini_topology(tmp_path: Path) -> Path:
    """Write a minimal step-01-style topology under tmp_path/topology/."""
    G = nx.Graph()
    for nid, isd, role in [
        (0, 0, "core"),
        (1, 0, "non-core"),
        (2, 0, "non-core"),
        (10, 1, "core"),
        (11, 1, "non-core"),
    ]:
        G.add_node(nid, isd=isd, role=role, x=float(nid), y=0.0)
    G.add_edge(0, 1, type="parent_child", latency=10.0, bandwidth=1000.0)
    G.add_edge(1, 2, type="parent_child", latency=8.0, bandwidth=1000.0)
    G.add_edge(10, 11, type="parent_child", latency=10.0, bandwidth=1000.0)
    G.add_edge(0, 10, type="core", latency=20.0, bandwidth=10000.0)
    G.add_edge(2, 11, type="peer", latency=15.0, bandwidth=5000.0)

    topo_dir = tmp_path / "topology"
    topo_dir.mkdir(parents=True)
    import pickle

    scion_topo = {
        "graph": G,
        "isds": [0, 1],
        "core_ases": {0, 10},
    }
    with open(topo_dir / "scion_topology.pkl", "wb") as f:
        pickle.dump(scion_topo, f)
    with open(topo_dir / "scion_topology.json", "w") as f:
        json.dump(
            {
                "isds": [0, 1],
                "core_ases": [0, 10],
                "graph": nx.node_link_data(G),
            },
            f,
        )
    return tmp_path


def test_load_topology_graph_from_pickle(tmp_path: Path):
    run = _mini_topology(tmp_path)
    G, core_ases, _ = load_topology_graph(run)
    assert G.number_of_nodes() == 5
    assert core_ases == {0, 10}


def test_discover_paths_uses_beacon_segments(tmp_path: Path):
    G, core_ases, _ = load_topology_graph(_mini_topology(tmp_path))
    sim = BeaconSimulator(max_segments_per_origin=200)
    segment_store, _ = sim.simulate(G, core_ases, tmp_path / "beacon_out")
    path_details, path_counts = discover_paths_for_topology(
        G, segment_store, core_ases
    )
    assert path_counts.get((2, 11), 0) >= 1
    for plist in path_details.values():
        for p in plist:
            seq = [h["as"] for h in p["hops"]]
            assert len(seq) == len(set(seq)), "path should not loop"


def test_run_beaconing_writes_artifacts(tmp_path: Path):
    run = _mini_topology(tmp_path)
    stats = run_beaconing(run)
    assert (run / "path_store.json").is_file()
    assert (run / "selected_pair.json").is_file()
    assert (run / "beacon_output" / "segments.json").is_file()
    assert stats["pair_pool_size"] >= 1
    sel = json.loads((run / "selected_pair.json").read_text())
    assert "pair_pool" in sel and sel["pair_pool_size"] >= 1
