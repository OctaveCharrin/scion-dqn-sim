"""Tests for ``src/beacon/beacon_sim_v2.CorrectedBeaconSimulator``.

Targets the new SCION-correctness invariants:

* Intra-ISD beaconing flows top-down only — children do not back-propagate
  PCBs to their parents.
* PEER edges never appear in beacon segments.
* Per-origin segment cap bounds fan-out.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from src.beacon.beacon_sim_v2 import CorrectedBeaconSimulator
from src.simulation.json_topology_adapter import json_topology_to_beacon_pickle


def _build_pickle(tmp_path: Path) -> Path:
    G = nx.Graph()
    for nid, isd, role, x, y in [
        (0, 0, "core", 0.0, 0.0),
        (1, 0, "non-core", 100.0, 0.0),
        (2, 0, "non-core", 200.0, 0.0),
        (10, 1, "core", 0.0, 100.0),
        (11, 1, "non-core", 100.0, 100.0),
    ]:
        G.add_node(nid, isd=isd, role=role, x=x, y=y)
    G.add_edge(0, 1, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(1, 2, type="PARENT_CHILD", latency=8.0, bandwidth=1000.0)
    G.add_edge(10, 11, type="PARENT_CHILD", latency=10.0, bandwidth=1000.0)
    G.add_edge(0, 10, type="CORE", latency=20.0, bandwidth=10000.0)
    G.add_edge(2, 11, type="PEER", latency=15.0, bandwidth=5000.0)

    topo = {
        "isds": [
            {"isd_id": 0, "member_ases": [0, 1, 2]},
            {"isd_id": 1, "member_ases": [10, 11]},
        ],
        "core_ases": [0, 10],
    }
    pkl = tmp_path / "topology.pkl"
    json_topology_to_beacon_pickle(topo, G, pkl)
    return pkl


def test_intra_beacons_are_top_down_only(tmp_path: Path):
    pkl = _build_pickle(tmp_path)
    sim = CorrectedBeaconSimulator(max_segments_per_origin=200)
    segments, _stats = sim.simulate(pkl, tmp_path)

    isd0_down = segments["down_segments_by_isd"].get(0, [])
    paths_isd0 = {tuple(s["path"]) for s in isd0_down}
    assert (0, 1) in paths_isd0
    assert (0, 1, 2) in paths_isd0

    # No down segment should originate at a non-core (intra-ISD beacons
    # originate only at cores).
    non_core_origins = [s for s in isd0_down if s["src"] in (1, 2)]
    assert non_core_origins == []


def test_peer_edges_not_used_for_beacons(tmp_path: Path):
    pkl = _build_pickle(tmp_path)
    sim = CorrectedBeaconSimulator(max_segments_per_origin=200)
    segments, _stats = sim.simulate(pkl, tmp_path)

    # PEER edge is between AS 2 (ISD 0) and AS 11 (ISD 1). No intra-ISD segment
    # should ever contain ASes from a different ISD.
    for key in ("down_segments_by_isd", "up_segments_by_isd"):
        for _isd, seg_list in segments[key].items():
            for seg in seg_list:
                p = seg["path"]
                assert not (2 in p and 11 in p), (
                    f"peer edge leaked into intra segment {p!r}"
                )

    # Core segments may only contain core ASes (the only core link is 0-10).
    for seg in segments["core_segments"]:
        p = seg["path"]
        assert all(node in (0, 10) for node in p)


def test_core_segments_are_directional_per_origin(tmp_path: Path):
    pkl = _build_pickle(tmp_path)
    sim = CorrectedBeaconSimulator(max_segments_per_origin=200)
    segments, _stats = sim.simulate(pkl, tmp_path)

    core_paths = {tuple(s["path"]) for s in segments["core_segments"]}
    # Both directions should be discovered (each core originates beacons).
    assert (0, 10) in core_paths
    assert (10, 0) in core_paths
