"""Tests for SCION path-builder behavior.

Targets the role-detection fallback (works without ``node['role']``) and
the intra-AS hop latency contribution.
"""

from __future__ import annotations

import networkx as nx

from src.simulation.path_builder import (
    INTRA_AS_HOP_LATENCY_MS,
    build_paths_for_pair,
    build_scion_paths_for_pair,
)


def _diamond() -> nx.Graph:
    G = nx.Graph()
    for n in range(4):
        G.add_node(n, isd=0, x=float(n), y=0.0)
    G.add_edge(0, 1, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(0, 2, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(1, 3, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(2, 3, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    return G


def test_build_paths_for_pair_includes_intra_as_hop_latency():
    G = _diamond()
    paths = build_paths_for_pair(G, 0, 3, max_paths=4)
    # All paths have 3 ASes (0 -> X -> 3) and total edge latency = 20.
    expected = 20.0 + INTRA_AS_HOP_LATENCY_MS * 3
    assert paths
    for p in paths:
        assert abs(p["static_metrics"]["total_latency"] - expected) < 1e-6
        assert p["static_metrics"]["hop_count"] == 2


def test_build_scion_paths_falls_through_when_no_role_attribute():
    """When neither ``node['role']`` nor ``core_ases`` flags a core endpoint,
    the SCION builder should still produce paths via the generic fallback.
    """
    G = _diamond()
    # No 'role' attribute; no core_ases passed.
    paths = build_scion_paths_for_pair(G, 0, 3, segment_store={"core": [], "up": {}, "down": {}})
    assert paths


def test_build_scion_paths_uses_core_ases_when_role_missing():
    """When the graph lacks ``role`` but ``core_ases`` is provided, the SCION
    builder uses it for role detection (matching what 02_run_beaconing now does).
    """
    G = nx.Graph()
    # Two ISDs, each with one core; PARENT_CHILD hierarchies; CORE link bridge.
    for nid, isd, x, y in [
        (0, 0, 0.0, 0.0),
        (1, 0, 1.0, 0.0),
        (2, 0, 2.0, 0.0),
        (10, 1, 0.0, 1.0),
        (11, 1, 1.0, 1.0),
    ]:
        G.add_node(nid, isd=isd, x=x, y=y)
    G.add_edge(0, 1, type="PARENT_CHILD", latency=5.0, bandwidth=1000.0)
    G.add_edge(1, 2, type="PARENT_CHILD", latency=5.0, bandwidth=1000.0)
    G.add_edge(0, 10, type="CORE", latency=20.0, bandwidth=10000.0)
    G.add_edge(10, 11, type="PARENT_CHILD", latency=5.0, bandwidth=1000.0)

    segment_store = {"core": [], "up": {}, "down": {}}
    paths = build_scion_paths_for_pair(
        G, 2, 11, segment_store=segment_store, core_ases={0, 10}
    )
    assert paths
