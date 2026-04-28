#!/usr/bin/env python3
"""
Run SCION beaconing simulation and discover candidate paths.

The pipeline now keeps **multiple AS pairs** in the path store so the DQN can
train and evaluate across pairs (not a single hand-picked one). The legacy
``selected_pair.json`` is still written for backwards compatibility with
downstream scripts that expected the "best" pair.
"""

import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx

from _common import resolve_run_dir, topology_dir

from src.beacon.beacon_sim_v2 import CorrectedBeaconSimulator
from src.simulation.json_topology_adapter import json_topology_to_beacon_pickle
from src.simulation.path_store import InMemoryPathStore
from src.simulation.path_builder import build_scion_paths_for_pair

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

# Prefer reusing the rich pickle written by step 1 (matches the JSON 1:1 and
# avoids drift if anything is added to the SCION topology dict). The pickle
# may still be missing in older runs, so fall through to JSON in that case.
pkl_candidates = [
    topology_dir(run_path) / "scion_topology.pkl",
    run_path / "scion_topology.pkl",
]
scion_topo = None
for cand in pkl_candidates:
    if cand.is_file():
        with open(cand, "rb") as f:
            scion_topo = pickle.load(f)
        break

if scion_topo is not None and isinstance(scion_topo.get("graph"), nx.Graph):
    G = scion_topo["graph"]
    core_ases = {int(x) for x in scion_topo.get("core_ases", set())}
else:
    G = nx.node_link_graph(topology_data["graph"])
    core_ases = {int(x) for x in topology_data.get("core_ases", []) or []}

print(f"\nLoaded topology with {G.number_of_nodes()} ASes")
print(f"  Core ASes: {len(core_ases)}")

# 1) Topology pickle for the beacon simulator (pandas nodes/edges).
beacon_pkl = run_path / "topology_beacon_input.pkl"
json_topology_to_beacon_pickle(topology_data, G, beacon_pkl)
print(f"Wrote beacon input topology: {beacon_pkl}")

# 2) Run SCION beacon simulation.
print("\nRunning SCION beacon simulation (CorrectedBeaconSimulator)...")
beacon_out = run_path / "beacon_output"
beacon_out.mkdir(parents=True, exist_ok=True)
simulator = CorrectedBeaconSimulator()
segment_store, _ = simulator.simulate(beacon_pkl, beacon_out)


def _enumerate_pairs():
    for src_as in G.nodes():
        for dst_as in G.nodes():
            if src_as == dst_as:
                continue
            yield int(src_as), int(dst_as)


print("\nAnalyzing path diversity...")
path_counts = defaultdict(int)
path_details = defaultdict(list)
n = G.number_of_nodes()

if n <= 200:
    for src_as, dst_as in _enumerate_pairs():
        paths = build_scion_paths_for_pair(
            G, int(src_as), int(dst_as), segment_store, core_ases=core_ases
        )
        if paths:
            path_counts[(src_as, dst_as)] = len(paths)
            path_details[(src_as, dst_as)] = paths
else:
    rng = random.Random(42)
    nodes = list(G.nodes())
    for _ in range(min(12000, 40 * n)):
        src_as, dst_as = rng.choice(nodes), rng.choice(nodes)
        if src_as == dst_as:
            continue
        paths = build_scion_paths_for_pair(
            G, int(src_as), int(dst_as), segment_store, core_ases=core_ases
        )
        if paths:
            path_counts[(src_as, dst_as)] = len(paths)
            path_details[(src_as, dst_as)] = paths

diverse_pairs = [(pair, c) for pair, c in path_counts.items() if c >= 5]
diverse_pairs.sort(key=lambda x: x[1], reverse=True)

print(f"\nTotal AS pairs with paths: {len(path_counts)}")
print(f"AS pairs with 5+ paths: {len(diverse_pairs)}")

if not path_counts:
    raise RuntimeError(
        "No paths found between any AS pair; topology may be disconnected."
    )

# Build the multi-pair path store. Keep all pairs for small topologies; for
# large topologies keep up to the top 200 most-diverse pairs (the DQN's action
# space scales with max(num_paths) across pairs, so capping diversity is fine).
print("\nBuilding path store...")
path_store = InMemoryPathStore()
if n <= 200:
    for (sa, da), plist in path_details.items():
        path_store.set_paths(sa, da, plist)
    pair_pool = list(path_details.keys())
else:
    by_count = sorted(path_counts.items(), key=lambda kv: kv[1], reverse=True)
    pair_pool = [pair for pair, _ in by_count[:200]]
    for pair in pair_pool:
        path_store.set_paths(int(pair[0]), int(pair[1]), path_details[pair])

# Pick a "best" pair for legacy single-pair consumers (and bookkeeping).
if diverse_pairs:
    best_pair, best_count = diverse_pairs[0]
else:
    best_pair, best_count = max(path_counts.items(), key=lambda x: x[1])
src_as, dst_as = best_pair
paths_for_selection = path_details[best_pair]

hop_counts = [len(p["hops"]) for p in paths_for_selection]
latencies = [
    p.get("static_metrics", {}).get("total_latency", 0.0) for p in paths_for_selection
]
bandwidths = [
    p.get("static_metrics", {}).get("min_bandwidth", 0.0) for p in paths_for_selection
]

print("\nSelected source-destination pair (legacy single-pair compat):")
print(f"  Source AS: {src_as}")
print(f"  Destination AS: {dst_as}")
print(f"  Number of paths: {best_count}")
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

# Pair pool used by training / evaluation: all pairs with paths in the store.
all_pair_counts = {
    f"{int(sa)}-{int(da)}": int(len(plist))
    for (sa, da), plist in path_details.items()
    if (sa, da) in pair_pool or n <= 200
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
print(f"\nSelected pair saved to: {selection_file}")

path_store_file = run_path / "path_store.pkl"
with open(path_store_file, "wb") as f:
    pickle.dump(path_store, f)
print(f"Path store saved to: {path_store_file}")

stats = {
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
    json.dump(stats, f, indent=2)

print(f"\nBeaconing statistics saved to: {stats_file}")
print("\nBeaconing simulation complete!")
