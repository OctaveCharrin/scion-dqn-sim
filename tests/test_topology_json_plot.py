"""Tests for topology JSON → PNG (merged into ``topology_visualizer``)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import networkx as nx

from src.visualization.topology_visualizer import (
    load_scion_topology_json,
    render_scion_topology_png,
)


def test_render_minimal_topology_png(tmp_path: Path):
    G = nx.DiGraph()
    G.add_node(1, x=0.0, y=0.0, isd=1)
    G.add_node(2, x=2.0, y=1.0, isd=1)
    G.add_node(3, x=1.0, y=3.0, isd=1)
    G.add_edge(1, 2, type="parent-child")
    G.add_edge(2, 1, type="child-parent")
    G.add_edge(2, 3, type="parent-child")
    G.add_edge(1, 3, type="PEER")

    payload = {
        "graph": nx.node_link_data(G),
        "core_ases": [1],
        "isds": [{"isd_id": 1, "member_ases": [1, 2, 3]}],
    }
    json_path = tmp_path / "scion_topology.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "topo.png"
    render_scion_topology_png(json_path, out)
    assert out.is_file()
    assert out.stat().st_size > 500


def test_load_roundtrip(tmp_path: Path):
    G = nx.Graph()
    G.add_node(10, x=0.0, y=0.0, isd=1)
    G.add_node(11, x=1.0, y=1.0, isd=1)
    G.add_edge(10, 11, type="core")
    payload = {"graph": nx.node_link_data(G), "core_ases": [10, 11], "isds": []}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    G2, core, _ = load_scion_topology_json(p)
    assert core == {10, 11}
    assert G2.number_of_nodes() == 2
