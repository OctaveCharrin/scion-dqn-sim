"""Enumerate simple paths on a topology graph for the evaluation path store.

Each emitted path's ``static_metrics.total_latency`` includes a small
**intra-AS forwarding contribution** in addition to the sum of edge latencies.
Real SCION ``ASEntry`` records carry intra-AS hop delay between ingress and
egress interfaces, so two paths that cross the same number of edges but
traverse a different set of ASes do not have identical static latency. Without
this, multiple SCION-shaped paths frequently tie on latency and the DQN's
action argmax becomes degenerate.
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx

# Constant intra-AS forwarding delay (ms). Small but non-zero so paths that
# traverse a different number of ASes are distinguishable on static metrics.
INTRA_AS_HOP_LATENCY_MS = 0.1


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
            base_latency = (
                sum(edge_latencies)
                if edge_latencies
                else float(hops[0].get("latency", 10.0))
            )
            total_latency = base_latency + INTRA_AS_HOP_LATENCY_MS * len(nodes)
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


def build_scion_paths_for_pair(
    G: nx.Graph,
    src: int,
    dst: int,
    segment_store: Dict[str, Any],
    max_paths: int = 30,
    core_ases: "set[int] | None" = None,
) -> List[Dict[str, Any]]:
    """Build SCION-shaped paths via Up→Core→Down segment composition.

    Uses the ``beacon_sim_v2`` segment store. Peer links and shortcuts
    between segments are detected from the graph and used to assemble
    additional paths.

    ``core_ases`` is optional but **highly recommended**: it lets the
    function detect a core src/dst even when the graph nodes lack a
    ``role`` attribute (older BRITE-generated graphs). Without it the
    function silently fell back to non-SCION ``shortest_simple_paths``
    whenever either endpoint was a core AS.

    Falls back to :func:`build_paths_for_pair` only when no SCION paths
    can be assembled (e.g. unreachable across segment store).
    """
    if src == dst or src not in G or dst not in G:
        return []

    if core_ases is None:
        core_ases = set()

    src_isd = G.nodes[src].get('isd', 0)
    dst_isd = G.nodes[dst].get('isd', 0)
    src_role = G.nodes[src].get('role')
    if src_role is None:
        src_role = 'core' if int(src) in core_ases else 'non-core'
    dst_role = G.nodes[dst].get('role')
    if dst_role is None:
        dst_role = 'core' if int(dst) in core_ases else 'non-core'
    
    # 1. Gather starting segments (Up or Virtual Core start/Down start)
    up_segs = []
    if src_role == 'core':
        up_segs.append({"path": [src], "dst": src})
    else:
        for s in segment_store.get('up_segments_by_isd', {}).get(src_isd, []):
            if s['src'] == src:
                up_segs.append(s)
                
    # 2. Gather ending segments (Down or Virtual Core end)
    down_segs = []
    if dst_role == 'core':
        down_segs.append({"path": [dst], "src": dst})
    else:
        for s in segment_store.get('down_segments_by_isd', {}).get(dst_isd, []):
            if s['dst'] == dst:
                down_segs.append(s)

    # 3. Gather Core segments (Core -> Core)
    core_segs = segment_store.get('core_segments', [])
    
    assembled_as_paths = []
    
    def add_path(seq):
        if len(seq) > 1 and seq[0] == src and seq[-1] == dst:
            assembled_as_paths.append(seq)

    for u_seg in up_segs:
        u_core = u_seg['dst']
        u_path = u_seg['path']
        
        for d_seg in down_segs:
            d_core = d_seg['src']
            d_path = d_seg['path']
            
            # Direct Up + Down combination (e.g. they share the identical ISDs/Cores)
            if u_core == d_core:
                seq = u_path + d_path[1:]
                add_path(seq)
                
            # Intersect via Core segments (Up + Core + Down)
            for c_seg in core_segs:
                if c_seg['src'] == u_core and c_seg['dst'] == d_core:
                    c_path = c_seg['path']
                    seq = u_path + c_path[1:] + d_path[1:]
                    add_path(seq)
                    
            # Peering combinations and early intersections (shortcuts)
            for i, u_node in enumerate(u_path):
                for j, d_node in enumerate(d_path):
                    if u_node == d_node and u_node != u_core:
                        # Shortcut if they intersect before the core (e.g. same parent AS)
                        seq = u_path[:i] + d_path[j:]
                        add_path(seq)
                    elif G.has_edge(u_node, d_node):
                        edge_data = G.get_edge_data(u_node, d_node) or {}
                        edge_type = str(edge_data.get("type", "")).lower()
                        if "peer" in edge_type:
                            seq = u_path[:i+1] + d_path[j:]
                            add_path(seq)
                    
    # Strict validation mapping
    unique_paths = []
    seen = set()
    for p in assembled_as_paths:
        tp = tuple(p)
        # Avoid looping SCION paths traversing the exact same node twice
        if tp not in seen and len(set(p)) == len(p):
            seen.add(tp)
            unique_paths.append(p)
            
    # Format to identically match build_paths_for_pair format dictionary requirement
    out = []
    for nodes in unique_paths:
        hops = []
        edge_latencies = []
        edge_bws = []
        
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
            
        base_latency = (
            sum(edge_latencies)
            if edge_latencies
            else float(hops[0].get("latency", 10.0))
        )
        total_latency = base_latency + INTRA_AS_HOP_LATENCY_MS * len(nodes)
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
        
    # Sort evaluated paths logically focusing on latency.
    out.sort(key=lambda x: x["static_metrics"]["total_latency"])
    
    # Optional Fallback immediately to generic generator if none are found by beaconing
    if not out:
        return build_paths_for_pair(G, src, dst, max_paths)
        
    return out[:max_paths]
