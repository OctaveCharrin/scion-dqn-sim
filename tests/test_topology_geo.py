"""Tests for ``src/topology/topology_geo`` shared helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.topology.topology_geo import (
    add_inter_isd_core_ring_edges,
    assign_isds_kmeans_coordinates,
    select_cores_by_centroid_proximity,
)
from src.topology.top_down_generator import TopDownSCIONGenerator


def test_kmeans_assigns_all_nodes() -> None:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1000, size=40)
    ys = rng.uniform(0, 1000, size=40)
    m = assign_isds_kmeans_coordinates(xs, ys, 4, random_state=42)
    assert len(m) == 40
    assert set(m.values()) <= {0, 1, 2, 3}


def test_inter_isd_ring_connects_cores() -> None:
    G = nx.Graph()
    for nid, (x, y, isd) in enumerate(
        [
            (0.0, 0.0, 0),
            (1.0, 0.0, 0),
            (100.0, 0.0, 1),
            (101.0, 0.0, 1),
        ]
    ):
        G.add_node(nid, x=x, y=y, isd=isd, role="core")
    isd_of = {i: G.nodes[i]["isd"] for i in range(4)}
    cores = {0, 1, 2, 3}
    G.add_edge(0, 1, type="CORE", latency=1.0, bandwidth=10_000.0)
    G.add_edge(2, 3, type="CORE", latency=1.0, bandwidth=10_000.0)
    n = add_inter_isd_core_ring_edges(G, isd_of, cores)
    assert n >= 1
    assert nx.is_connected(G)


@pytest.mark.parametrize("seed", range(15))
def test_top_down_global_connectivity(seed: int) -> None:
    d = TopDownSCIONGenerator(seed=seed).generate(n_isds=5, n_nodes=80)
    assert nx.is_connected(d["graph"])
