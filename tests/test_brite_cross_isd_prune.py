"""BRITE converter cross-ISD non-core edge pruning."""

from __future__ import annotations

import networkx as nx
import numpy as np
from src.topology.brite2scion_converter import BRITE2SCIONConverter


def test_prune_removes_expected_fraction() -> None:
    conv = BRITE2SCIONConverter()
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, x=float(i), y=0.0, isd=0 if i < 3 else 1)
    core = {0, 3}
    isd = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    # Cross-ISD non-core: (1,4), (1,5), (2,4), (2,5) — 4 edges
    G.add_edge(1, 4, delay=1.0, bandwidth=10.0)
    G.add_edge(1, 5, delay=1.0, bandwidth=10.0)
    G.add_edge(2, 4, delay=1.0, bandwidth=10.0)
    G.add_edge(2, 5, delay=1.0, bandwidth=10.0)
    # Same-ISD non-core (should never be removed)
    G.add_edge(1, 2, delay=1.0, bandwidth=10.0)
    rng = np.random.default_rng(0)
    n = conv._prune_cross_isd_noncore_edges(G, core, isd, 0.5, rng)
    assert n == 2
    assert G.has_edge(1, 2)
    assert G.number_of_edges() == 3  # 1 same-ISD + 2 cross remaining
