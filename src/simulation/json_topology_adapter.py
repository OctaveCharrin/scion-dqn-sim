"""
Convert evaluation ``scion_topology.json`` (NetworkX node-link) into the
``topology.pkl`` shape expected by ``beacon_sim_v2.CorrectedBeaconSimulator``.

Direction handling
------------------

SCION intra-ISD beaconing flows strictly **top-down**: a core AS originates
PCBs that propagate **parent → child** along the ISD hierarchy. The beacon
simulator distinguishes the two directions of a hierarchy edge as
``parent-child`` (forward, from parent toward child) and ``child-parent``
(reverse, from child up to parent). Because the JSON topology graph is
undirected, this adapter recomputes the parent/child orientation from
each ISD's distance-to-core BFS and emits the two directed rows
accordingly.

PEER edges are emitted with ``type='peer'`` in both directions; the beacon
simulator skips them during PCB propagation (peering is consumed by the
path-builder as a shortcut, not a beacon carrier).
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set

import networkx as nx
import pandas as pd


def _edge_type_for_beacon(
    u: int, v: int, raw_type: str, core_ases: Set[int]
) -> str:
    """Map arbitrary link types to ``beacon_sim_v2`` edge families.

    Returned strings are exactly what the beacon simulator expects:
    ``core``, ``parent-child``, ``peer``. PEER edges are emitted as ``peer``
    so the beacon simulator can skip them during PCB propagation (they are
    only consumed by ``path_builder`` as shortcut links).
    """
    uc = u in core_ases
    vc = v in core_ases
    t = (raw_type or "").upper().replace("-", "_")
    if "PEER" in t or "PEERING" in t:
        return "peer"
    if uc and vc:
        return "core"
    if "CORE" in t:
        return "core"
    return "parent-child"


def _distance_to_core_per_isd(
    graph: nx.Graph,
    isd_of: Dict[int, int],
    core_ases: Set[int],
) -> Dict[int, int]:
    """Multi-source BFS within each ISD from its core ASes.

    Returns a node→distance mapping. Nodes unreachable from any same-ISD core
    receive a sentinel large distance.
    """
    SENTINEL = 10**9
    dist: Dict[int, int] = {int(n): SENTINEL for n in graph.nodes}
    cores_by_isd: Dict[int, List[int]] = {}
    for c in core_ases:
        cores_by_isd.setdefault(int(isd_of[int(c)]), []).append(int(c))

    for isd, cores in cores_by_isd.items():
        queue: deque = deque()
        for c in cores:
            dist[int(c)] = 0
            queue.append(int(c))
        while queue:
            u = queue.popleft()
            for v in graph.neighbors(u):
                if int(isd_of.get(int(v), -1)) != isd:
                    continue
                if dist[int(v)] != SENTINEL:
                    continue
                dist[int(v)] = dist[u] + 1
                queue.append(int(v))
    return dist


def _is_peer_edge(raw_type: str) -> bool:
    t = (raw_type or "").upper().replace("-", "_")
    return "PEER" in t or "PEERING" in t


def json_topology_to_beacon_pickle(
    topology_data: Dict[str, Any],
    graph: nx.Graph,
    output_pkl: Path,
) -> Path:
    """
    Build ``{'nodes': DataFrame, 'edges': DataFrame, 'metadata': ...}`` and pickle it.

    ``topology_data`` should be the JSON dict (``isds``, ``core_ases``, ``graph`` key
    for node-link). ``graph`` should be the reconstructed NetworkX graph with node
    ``isd`` and edge attributes.

    The output edge DataFrame contains TWO rows per undirected edge (one per
    direction) so the beacon simulator can distinguish ``parent-child`` from
    ``child-parent`` traversal without consulting the original orientation.
    """
    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    core_raw = topology_data.get("core_ases") or []
    core_ases = {int(x) for x in core_raw}

    rows = []
    isd_of: Dict[int, int] = {}
    for nid in graph.nodes:
        nattr = graph.nodes[nid]
        isd = int(nattr.get("isd", 0))
        isd_of[int(nid)] = isd
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

    dist_to_core = _distance_to_core_per_isd(graph, isd_of, core_ases)

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
        u_core = u in core_ases
        v_core = v in core_ases
        u_isd = isd_of.get(u)
        v_isd = isd_of.get(v)
        same_isd = u_isd == v_isd

        u_if = int(data.get("src_if", data.get("u_if", if_cnt)))
        v_if = int(data.get("dst_if", data.get("v_if", if_cnt + 1)))
        if_cnt += 2
        lat = float(data.get("latency", data.get("delay", 10.0)))
        bw = float(data.get("bandwidth", data.get("capacity", 1000.0)))

        # Decide canonical parent/child orientation for hierarchy edges.
        # ``parent`` is the AS closer to a same-ISD core; ``child`` is farther.
        is_peer = _is_peer_edge(raw_type) or not same_isd and not (u_core and v_core)
        if is_peer:
            et_uv = "peer"
            et_vu = "peer"
            parent, child = None, None
        elif u_core and v_core:
            et_uv = "core"
            et_vu = "core"
            parent, child = None, None
        else:
            d_u = dist_to_core.get(u, 10**9)
            d_v = dist_to_core.get(v, 10**9)
            if d_u < d_v:
                parent, child = u, v
            elif d_v < d_u:
                parent, child = v, u
            else:
                # Same depth: stable tie-break by AS id.
                parent, child = (u, v) if u < v else (v, u)

            # Forward (parent→child) gets ``parent-child``;
            # reverse (child→parent) gets ``child-parent``.
            if parent == u:
                et_uv, et_vu = "parent-child", "child-parent"
            else:
                et_uv, et_vu = "child-parent", "parent-child"

        common = {
            "type_uv": et_uv,
            "type_vu": et_vu,
            "latency": lat,
            "capacity": bw,
        }
        erows.append(
            {
                "u": u,
                "v": v,
                "u_if": u_if,
                "v_if": v_if,
                "type": common["type_uv"],
                "latency": lat,
                "capacity": bw,
            }
        )
        erows.append(
            {
                "u": v,
                "v": u,
                "u_if": v_if,
                "v_if": u_if,
                "type": common["type_vu"],
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
