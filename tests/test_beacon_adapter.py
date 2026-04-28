"""Tests for the JSON → beacon-pickle adapter.

These tests focus on SCION-correctness invariants:

* PEER edges are emitted as ``type='peer'`` (both directions).
* Each undirected hierarchy edge is split into a ``parent-child`` and a
  ``child-parent`` directed row, with the parent identified by lower
  distance-to-core.
* Core-to-core edges keep ``type='core'`` in both directions.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from src.simulation.json_topology_adapter import (
    _distance_to_core_per_isd,
    _edge_type_for_beacon,
    json_topology_to_beacon_pickle,
)


def _build_graph() -> nx.Graph:
    """Two-ISD topology with a clear core/non-core hierarchy.

    ISD 0: core = 0; non-core = 1, 2 (1 directly attached to core, 2 via 1).
    ISD 1: core = 10; non-core = 11.
    Cross-ISD: a CORE link 0-10 and a PEER link 2-11.
    """
    G = nx.Graph()
    for nid, isd, x, y in [
        (0, 0, 0.0, 0.0),
        (1, 0, 100.0, 0.0),
        (2, 0, 200.0, 0.0),
        (10, 1, 0.0, 100.0),
        (11, 1, 100.0, 100.0),
    ]:
        role = "core" if nid in (0, 10) else "non-core"
        G.add_node(nid, isd=isd, x=x, y=y, role=role)

    G.add_edge(0, 1, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(1, 2, type="PARENT_CHILD", latency=8.0, bandwidth=1000.0)
    G.add_edge(10, 11, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(0, 10, type="CORE", latency=20.0, bandwidth=10000.0)
    G.add_edge(2, 11, type="PEER", latency=15.0, bandwidth=5000.0)
    return G


def test_edge_type_for_beacon_maps_peer_to_peer():
    cores = {0, 10}
    assert _edge_type_for_beacon(2, 11, "PEER", cores) == "peer"
    assert _edge_type_for_beacon(2, 11, "PEERING", cores) == "peer"


def test_edge_type_for_beacon_keeps_core_for_core_to_core():
    cores = {0, 10}
    assert _edge_type_for_beacon(0, 10, "CORE", cores) == "core"


def test_distance_to_core_per_isd_uses_cores_as_sources():
    G = _build_graph()
    isd_of = {n: G.nodes[n]["isd"] for n in G.nodes}
    dist = _distance_to_core_per_isd(G, isd_of, {0, 10})
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[2] == 2
    assert dist[10] == 0
    assert dist[11] == 1


def test_adapter_emits_directed_rows_with_correct_orientation(tmp_path: Path):
    G = _build_graph()
    topo = {
        "isds": [
            {"isd_id": 0, "member_ases": [0, 1, 2]},
            {"isd_id": 1, "member_ases": [10, 11]},
        ],
        "core_ases": [0, 10],
    }
    out = tmp_path / "beacon.pkl"
    json_topology_to_beacon_pickle(topo, G, out)
    with open(out, "rb") as f:
        bundle = pickle.load(f)
    edges: pd.DataFrame = bundle["edges"]

    def _row(u: int, v: int) -> pd.Series:
        match = edges[(edges["u"] == u) & (edges["v"] == v)]
        assert not match.empty, f"missing directed row {u}->{v}"
        return match.iloc[0]

    # Hierarchy edge 0-1: 0 is core (parent), 1 is child.
    assert _row(0, 1)["type"] == "parent-child"
    assert _row(1, 0)["type"] == "child-parent"

    # Hierarchy edge 1-2: 1 is closer to core 0 than 2.
    assert _row(1, 2)["type"] == "parent-child"
    assert _row(2, 1)["type"] == "child-parent"

    # Core link 0-10: ``core`` in both directions.
    assert _row(0, 10)["type"] == "core"
    assert _row(10, 0)["type"] == "core"

    # PEER link 2-11: ``peer`` in both directions.
    assert _row(2, 11)["type"] == "peer"
    assert _row(11, 2)["type"] == "peer"


def test_adapter_round_trips_node_role(tmp_path: Path):
    G = _build_graph()
    topo = {
        "isds": [
            {"isd_id": 0, "member_ases": [0, 1, 2]},
            {"isd_id": 1, "member_ases": [10, 11]},
        ],
        "core_ases": [0, 10],
    }
    out = tmp_path / "beacon.pkl"
    json_topology_to_beacon_pickle(topo, G, out)
    bundle = pickle.loads(out.read_bytes())
    nodes: pd.DataFrame = bundle["nodes"]
    role_for = dict(zip(nodes["as_id"], nodes["role"]))
    assert role_for[0] == "core"
    assert role_for[10] == "core"
    assert role_for[1] == "non-core"
    assert role_for[2] == "non-core"
