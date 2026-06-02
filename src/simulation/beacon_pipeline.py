"""Beaconing and path discovery for evaluation run directories (pipeline step 02)."""

from __future__ import annotations

import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from src.beacon.beacon_sim import BeaconSimulator
from src.simulation.path_builder import build_scion_paths_for_pair
from src.simulation.path_store import InMemoryPathStore
from src.simulation.run_context import topology_dir

# Cap pair enumeration on large graphs (full mesh is expensive).
MAX_NODES_FULL_PAIR_SCAN = 200
LARGE_TOPO_SAMPLE_PAIRS = 40  # multiplier: min(12000, 40 * n)


def load_topology_graph(run_path: Path) -> Tuple[nx.Graph, set[int], Dict[str, Any]]:
    """Load the SCION topology graph and core AS set from step 01 artifacts."""
    run_path = Path(run_path)
    topo_dir = topology_dir(run_path)
    pkl_path = topo_dir / "scion_topology.pkl"
    json_path = topo_dir / "scion_topology.json"

    if pkl_path.is_file():
        with open(pkl_path, "rb") as f:
            scion_topo = pickle.load(f)
        G = scion_topo["graph"]
        if not isinstance(G, nx.Graph):
            raise TypeError(f"Expected NetworkX graph in {pkl_path}")
        core_ases = {int(x) for x in scion_topo.get("core_ases", set())}
        topology_data = {
            "isds": scion_topo.get("isds", []),
            "core_ases": list(core_ases),
            "graph": nx.node_link_data(G),
        }
        return G, core_ases, topology_data

    if not json_path.is_file():
        raise FileNotFoundError(
            f"Missing topology under {topo_dir}. Run 01_generate_topology.py first."
        )
    with open(json_path, "r") as f:
        topology_data = json.load(f)
    G = nx.node_link_graph(topology_data["graph"])
    core_ases = {int(x) for x in topology_data.get("core_ases", []) or []}
    return G, core_ases, topology_data


def _enumerate_all_pairs(G: nx.Graph):
    for src_as in G.nodes():
        for dst_as in G.nodes():
            if src_as != dst_as:
                yield int(src_as), int(dst_as)


def discover_paths_for_topology(
    G: nx.Graph,
    segment_store: Dict[str, Any],
    core_ases: set[int],
    *,
    max_nodes_full_scan: int = MAX_NODES_FULL_PAIR_SCAN,
) -> Tuple[Dict[Tuple[int, int], List[Dict]], Dict[Tuple[int, int], int]]:
    """Build SCION paths for all (or sampled) AS pairs using the segment store."""
    path_details: Dict[Tuple[int, int], List[Dict]] = {}
    path_counts: Dict[Tuple[int, int], int] = {}
    n = G.number_of_nodes()

    if n <= max_nodes_full_scan:
        pairs = _enumerate_all_pairs(G)
    else:
        rng = random.Random(42)
        nodes = list(G.nodes())
        pairs = []
        for _ in range(min(12000, LARGE_TOPO_SAMPLE_PAIRS * n)):
            src_as, dst_as = int(rng.choice(nodes)), int(rng.choice(nodes))
            if src_as != dst_as:
                pairs.append((src_as, dst_as))

    for src_as, dst_as in pairs:
        paths = build_scion_paths_for_pair(
            G, src_as, dst_as, segment_store, core_ases=core_ases
        )
        if paths:
            path_details[(src_as, dst_as)] = paths
            path_counts[(src_as, dst_as)] = len(paths)

    return path_details, path_counts


def run_beaconing(run_path: Path) -> Dict[str, Any]:
    """Run SCION beacon simulation and write path-store artifacts.

    Uses :class:`~src.beacon.beacon_sim.BeaconSimulator` (core mesh + top-down
    intra-ISD PCBs; peer links excluded from beacon propagation).

    Writes under ``run_path``:

    * ``beacon_output/segments.json``, ``beacon_output/beacon_stats.json``
    * ``path_store.json``, ``selected_pair.json``, ``beaconing_stats.json``
    """
    run_path = Path(run_path)
    G, core_ases, _topology_data = load_topology_graph(run_path)

    print(f"\nLoaded topology with {G.number_of_nodes()} ASes")
    print(f"  Core ASes: {len(core_ases)}")

    beacon_out = run_path / "beacon_output"
    print("\nRunning SCION beacon simulation (BeaconSimulator)...")
    simulator = BeaconSimulator()
    segment_store, beacon_stats = simulator.simulate(G, core_ases, beacon_out)

    print("\nDiscovering paths from beacon segments (Up → Core → Down)...")
    path_details, path_counts = discover_paths_for_topology(
        G, segment_store, core_ases
    )

    diverse_pairs = sorted(
        [(pair, c) for pair, c in path_counts.items() if c >= 5],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\nTotal AS pairs with paths: {len(path_counts)}")
    print(f"AS pairs with 5+ paths: {len(diverse_pairs)}")

    if not path_counts:
        raise RuntimeError(
            "No paths found between any AS pair; check topology connectivity "
            "and beacon segment coverage."
        )

    n = G.number_of_nodes()
    path_store = InMemoryPathStore()
    if n <= MAX_NODES_FULL_PAIR_SCAN:
        pair_pool = list(path_details.keys())
        for pair, plist in path_details.items():
            path_store.set_paths(pair[0], pair[1], plist)
    else:
        by_count = sorted(path_counts.items(), key=lambda kv: kv[1], reverse=True)
        pair_pool = [pair for pair, _ in by_count[:200]]
        for pair in pair_pool:
            path_store.set_paths(int(pair[0]), int(pair[1]), path_details[pair])

    if diverse_pairs:
        best_pair, best_count = diverse_pairs[0]
    else:
        best_pair, best_count = max(path_counts.items(), key=lambda x: x[1])
    src_as, dst_as = best_pair
    paths_for_selection = path_details[best_pair]

    hop_counts = [len(p["hops"]) for p in paths_for_selection]
    latencies = [
        p.get("static_metrics", {}).get("total_latency", 0.0)
        for p in paths_for_selection
    ]
    bandwidths = [
        p.get("static_metrics", {}).get("min_bandwidth", 0.0)
        for p in paths_for_selection
    ]

    print("\nReference pair (highest path diversity):")
    print(f"  Source AS: {src_as}")
    print(f"  Destination AS: {dst_as}")
    print(f"  Number of paths: {best_count}")
    if hop_counts:
        print(
            f"  Hop counts: min={min(hop_counts)}, max={max(hop_counts)}, "
            f"avg={np.mean(hop_counts):.1f}"
        )
        print(
            f"  Latencies (ms): min={min(latencies):.1f}, max={max(latencies):.1f}, "
            f"avg={np.mean(latencies):.1f}"
        )

    all_pair_counts = {
        f"{int(sa)}-{int(da)}": int(len(plist))
        for (sa, da), plist in path_details.items()
        if (sa, da) in pair_pool or n <= MAX_NODES_FULL_PAIR_SCAN
    }
    max_num_paths = max(all_pair_counts.values()) if all_pair_counts else 0

    selection = {
        "source_as": int(src_as),
        "destination_as": int(dst_as),
        "num_paths": int(best_count),
        "path_metrics": {
            "hop_counts": hop_counts,
            "latencies": latencies,
            "bandwidths": bandwidths,
        },
        "pair_pool": [[int(sa), int(da)] for (sa, da) in pair_pool],
        "pair_pool_size": len(pair_pool),
        "max_num_paths": int(max_num_paths),
    }

    selection_file = run_path / "selected_pair.json"
    with open(selection_file, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"\nWrote {selection_file}")

    path_store_file = run_path / "path_store.json"
    path_store.save(str(path_store_file))
    print(f"Wrote {path_store_file}")

    stats = {
        "beacon_simulation": beacon_stats,
        "total_as_pairs": len(path_counts),
        "pairs_with_paths": sum(1 for c in path_counts.values() if c > 0),
        "pairs_with_5plus_paths": len(diverse_pairs),
        "pair_pool_size": len(pair_pool),
        "max_paths_for_pair": max(path_counts.values()) if path_counts else 0,
        "avg_paths_per_pair": float(np.mean(list(path_counts.values())))
        if path_counts
        else 0.0,
        "path_distribution": dict(Counter(path_counts.values())),
    }
    stats_file = run_path / "beaconing_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Wrote {stats_file}")

    return stats
