"""
Convert evaluation ``scion_topology.json`` (NetworkX node-link) into the
``topology.pkl`` shape expected by ``beacon_sim_v2.CorrectedBeaconSimulator``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pandas as pd


def _edge_type_for_beacon(u: int, v: int, raw_type: str, core_ases: Set[int]) -> str:
    """Map arbitrary link types to beacon_sim_v2 edge families."""
    uc = u in core_ases
    vc = v in core_ases
    if uc and vc:
        return "core"
    t = (raw_type or "").upper().replace("-", "_")
    if "PEER" in t or "PEERING" in t:
        return "parent-child"
    if "CORE" in t:
        return "core"
    if uc or vc:
        return "parent-child"
    return "parent-child"


def json_topology_to_beacon_pickle(
    topology_data: Dict[str, Any],
    graph: nx.Graph,
    output_pkl: Path,
) -> Path:
    """
    Build ``{'nodes': DataFrame, 'edges': DataFrame, 'metadata': ...}`` and pickle it.

    ``topology_data`` should be the JSON dict (``isds``, ``core_ases``, ``graph`` key for node-link).
    ``graph`` should be the reconstructed NetworkX graph with node ``isd`` and edge attributes.
    """
    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    core_raw = topology_data.get("core_ases") or []
    core_ases = {int(x) for x in core_raw}

    rows = []
    for nid in graph.nodes:
        nattr = graph.nodes[nid]
        isd = int(nattr.get("isd", 0))
        role = "core" if int(nid) in core_ases else "non-core"
        rows.append(
            {
                "as_id": int(nid),
                "isd": isd,
                "role": role,
                "x": float(nattr.get("x", 0.0)),
                "y": float(nattr.get("y", 0.0)),
                "degree": int(graph.degree(nid)),
            }
        )
    node_df = pd.DataFrame(rows)

    erows: List[Dict[str, Any]] = []
    if_cnt = 0
    seen = set()
    for u, v, data in graph.edges(data=True):
        u, v = int(u), int(v)
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
        raw_type = str(data.get("type", ""))
        et = _edge_type_for_beacon(u, v, raw_type, core_ases)
        lat = float(data.get("latency", data.get("delay", 10.0)))
        bw = float(data.get("bandwidth", data.get("capacity", 1000.0)))
        u_if = int(data.get("src_if", data.get("u_if", if_cnt)))
        v_if = int(data.get("dst_if", data.get("v_if", if_cnt + 1)))
        if_cnt += 2
        row = {
            "u": u,
            "v": v,
            "u_if": u_if,
            "v_if": v_if,
            "type": et,
            "latency": lat,
            "capacity": bw,
        }
        erows.append(row)
        erows.append(
            {
                "u": v,
                "v": u,
                "u_if": v_if,
                "v_if": u_if,
                "type": et,
                "latency": lat,
                "capacity": bw,
            }
        )
    edge_df = pd.DataFrame(erows)

    topology = {
        "nodes": node_df,
        "edges": edge_df,
        "metadata": {
            "n_nodes": len(node_df),
            "n_edges": len(edge_df),
            "n_isds": int(node_df["isd"].nunique()) if len(node_df) else 0,
            "n_core_ases": len(core_ases),
        },
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(topology, f)

    return output_pkl
