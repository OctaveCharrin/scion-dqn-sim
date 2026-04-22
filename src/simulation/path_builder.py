"""Enumerate simple paths on a topology graph for the evaluation path store."""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx


def _edge_metrics(G: nx.Graph, u: int, v: int) -> tuple:
    data = G.get_edge_data(u, v) or {}
    lat = float(data.get("latency", data.get("delay", 10.0)))
    bw = float(data.get("bandwidth", data.get("capacity", 1000.0)))
    return lat, bw


def build_paths_for_pair(
    G: nx.Graph,
    src: int,
    dst: int,
    max_paths: int = 30,
    max_cutoff: int = 12,
) -> List[Dict[str, Any]]:
    """
    Build path dicts compatible with ``03_simulate_traffic`` / evaluation env / baselines.

    Each path has ``hops`` (list of per-node dicts with ``as``, ``latency``, ``bandwidth``)
    and ``static_metrics`` with ``hop_count``, ``total_latency``, ``min_bandwidth``.
    """
    if src == dst or src not in G or dst not in G:
        return []
    cutoff = min(max_cutoff, max(2, G.number_of_nodes()))
    out: List[Dict[str, Any]] = []
    try:
        gen = nx.shortest_simple_paths(G, src, dst)
        for nodes in gen:
            if len(nodes) - 1 > cutoff:
                continue
            hops: List[Dict[str, Any]] = []
            edge_latencies: List[float] = []
            edge_bws: List[float] = []
            for i, asn in enumerate(nodes):
                hops.append({"as": int(asn), "latency": 0.0, "bandwidth": float("inf")})
            for i in range(len(nodes) - 1):
                u, v = int(nodes[i]), int(nodes[i + 1])
                lat, bw = _edge_metrics(G, u, v)
                edge_latencies.append(lat)
                edge_bws.append(bw)
                hops[i]["latency"] = lat
                hops[i]["bandwidth"] = bw
            if nodes:
                hops[-1]["latency"] = 0.0
                hops[-1]["bandwidth"] = min(edge_bws) if edge_bws else 1000.0
            total_latency = sum(edge_latencies) if edge_latencies else float(
                hops[0].get("latency", 10.0)
            )
            min_bw = min(edge_bws) if edge_bws else 1000.0
            out.append(
                {
                    "hops": hops,
                    "static_metrics": {
                        "hop_count": max(1, len(nodes) - 1),
                        "total_latency": float(total_latency),
                        "min_bandwidth": float(min_bw),
                    },
                }
            )
            if len(out) >= max_paths:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return out
