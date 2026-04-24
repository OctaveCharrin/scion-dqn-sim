#!/usr/bin/env python3
"""
Run SCION beaconing simulation and analyze path diversity
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx

from _common import resolve_run_dir, topology_dir

from src.beacon.beacon_sim_v2 import CorrectedBeaconSimulator
from src.simulation.json_topology_adapter import json_topology_to_beacon_pickle
from src.simulation.path_store import InMemoryPathStore
from src.simulation.path_builder import build_paths_for_pair

run_dir = resolve_run_dir()
run_path = Path(run_dir)

# Load topology JSON
topology_file = topology_dir(run_path) / "scion_topology.json"
if not topology_file.is_file():
    legacy = run_path / "scion_topology.json"
    if legacy.is_file():
        topology_file = legacy
with open(topology_file, "r") as f:
    topology_data = json.load(f)

G = nx.node_link_graph(topology_data["graph"])

print(f"\nLoaded topology with {G.number_of_nodes()} ASes")

# 1) Topology pickle for BRITE beacon simulator (pandas nodes/edges)
beacon_pkl = run_path / "topology_beacon_input.pkl"
json_topology_to_beacon_pickle(topology_data, G, beacon_pkl)
print(f"Wrote beacon input topology: {beacon_pkl}")

# 2) Run corrected beacon simulation (writes segments_corrected.pkl, etc.)
print("\nRunning SCION beacon simulation (CorrectedBeaconSimulator)...")
beacon_out = run_path / "beacon_output"
beacon_out.mkdir(parents=True, exist_ok=True)
simulator = CorrectedBeaconSimulator()
simulator.simulate(beacon_pkl, beacon_out)

# 3) Analyze path diversity and pick (src, dst) using graph enumeration
print("\nAnalyzing path diversity...")
path_counts = defaultdict(int)
path_details = defaultdict(list)
n = G.number_of_nodes()

def _enumerate_pairs():
    for src_as in G.nodes():
        for dst_as in G.nodes():
            if src_as == dst_as:
                continue
            yield int(src_as), int(dst_as)


if n <= 200:
    for src_as, dst_as in _enumerate_pairs():
        paths = build_paths_for_pair(G, src_as, dst_as)
        if paths:
            path_counts[(src_as, dst_as)] = len(paths)
            path_details[(src_as, dst_as)] = paths
else:
    import random

    rng = random.Random(42)
    nodes = list(G.nodes())
    for _ in range(min(12000, 40 * n)):
        src_as, dst_as = rng.choice(nodes), rng.choice(nodes)
        if src_as == dst_as:
            continue
        paths = build_paths_for_pair(G, int(src_as), int(dst_as))
        if paths:
            path_counts[(src_as, dst_as)] = len(paths)
            path_details[(src_as, dst_as)] = paths

diverse_pairs = [(pair, c) for pair, c in path_counts.items() if c >= 5]
diverse_pairs.sort(key=lambda x: x[1], reverse=True)

print(f"\nTotal AS pairs with paths: {len(path_counts)}")
print(f"AS pairs with 5+ paths: {len(diverse_pairs)}")

hop_counts = []
latencies = []
bandwidths = []
paths_for_selection: list = []

if diverse_pairs:
    best_pair, best_count = diverse_pairs[0]
    src_as, dst_as = best_pair
    print("\nSelected source-destination pair:")
    print(f"  Source AS: {src_as}")
    print(f"  Destination AS: {dst_as}")
    print(f"  Number of paths: {best_count}")
    paths_for_selection = path_details[best_pair]
    for path in paths_for_selection:
        hop_counts.append(len(path["hops"]))
        latencies.append(sum(hop.get("latency", 10) for hop in path["hops"]))
        bandwidths.append(min(hop.get("bandwidth", 1000) for hop in path["hops"]))
    if hop_counts:
        print("\n  Path characteristics:")
        print(
            f"    Hop counts: min={min(hop_counts)}, max={max(hop_counts)}, avg={np.mean(hop_counts):.1f}"
        )
        print(
            f"    Latencies (ms): min={min(latencies):.1f}, max={max(latencies):.1f}, avg={np.mean(latencies):.1f}"
        )
        print(
            f"    Bandwidths (Mbps): min={min(bandwidths):.1f}, max={max(bandwidths):.1f}, avg={np.mean(bandwidths):.1f}"
        )
    selection = {
        "source_as": int(src_as),
        "destination_as": int(dst_as),
        "num_paths": best_count,
        "path_metrics": {
            "hop_counts": hop_counts,
            "latencies": latencies,
            "bandwidths": bandwidths,
        },
    }
else:
    print("\nWarning: No AS pairs with 5+ paths found!")
    print("Selecting pair with most paths...")
    if not path_counts:
        raise RuntimeError("No paths found between any AS pair; topology may be disconnected.")
    best_pair, count = max(path_counts.items(), key=lambda x: x[1])
    src_as, dst_as = best_pair
    paths_for_selection = path_details[best_pair]
    selection = {
        "source_as": int(src_as),
        "destination_as": int(dst_as),
        "num_paths": count,
    }

selection_file = run_path / "selected_pair.json"
with open(selection_file, "w") as f:
    json.dump(selection, f, indent=2)
print(f"\nSelected pair saved to: {selection_file}")

# Build path store: full graph for small topologies; selected pair only for large graphs
print("\nBuilding path store...")
path_store = InMemoryPathStore()
if n <= 200:
    for (sa, da), plist in path_details.items():
        path_store.set_paths(sa, da, plist)
else:
    full_paths = build_paths_for_pair(G, int(src_as), int(dst_as), max_paths=50)
    path_store.set_paths(int(src_as), int(dst_as), full_paths)
    selection["num_paths"] = len(full_paths)

path_store_file = run_path / "path_store.pkl"
with open(path_store_file, "wb") as f:
    pickle.dump(path_store, f)
print(f"Path store saved to: {path_store_file}")

if n > 200:
    with open(selection_file, "w") as f:
        json.dump(selection, f, indent=2)

stats = {
    "total_as_pairs": len(path_counts),
    "pairs_with_paths": sum(1 for c in path_counts.values() if c > 0),
    "pairs_with_5plus_paths": len(diverse_pairs),
    "max_paths_for_pair": max(path_counts.values()) if path_counts else 0,
    "avg_paths_per_pair": float(np.mean(list(path_counts.values()))) if path_counts else 0.0,
    "path_distribution": dict(Counter(path_counts.values())),
}

stats_file = run_path / "beaconing_stats.json"
with open(stats_file, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nBeaconing statistics saved to: {stats_file}")
print("\nBeaconing simulation complete!")
