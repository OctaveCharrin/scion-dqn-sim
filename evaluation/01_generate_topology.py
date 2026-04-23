#!/usr/bin/env python3
"""
Generate dense SCION topology using BRITE.

All BRITE / SCION topology artifacts (config, ``.brite``, JSON, pickle, step
PNGs) are written under ``<run_dir>/topology/``.
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx

from _common import topology_dir
from src.topology.brite_cfg_gen import BRITEConfigGenerator, run_brite
from src.topology.brite2scion_converter import BRITE2SCIONConverter


# Allow this step to create a new run directory if none is provided.
if len(sys.argv) > 1:
    run_dir = sys.argv[1]
    print(f"Using run directory: {run_dir}")
else:
    run_dir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Using run directory: {run_dir}")

topo_dir = topology_dir(run_dir)
topo_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating dense SCION topology under {topo_dir}")

# Locate the BRITE distribution relative to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
BRITE_PATH = REPO_ROOT / "external" / "brite"

# Step 1: Generate dense BRITE topology configuration
print("\n1. Generating BRITE configuration...")
brite_gen = BRITEConfigGenerator()

# Configure for dense topology (numeric keys match BRITE ModelConstants / parser)
# Set EVAL_BRITE_N_NODES for a faster smoke test (e.g. 45); default is large-scale.
_eval_n = os.environ.get("EVAL_BRITE_N_NODES", "").strip()
_default_nodes = int(_eval_n) if _eval_n.isdigit() else 1000
config_params = {
    "n_nodes": _default_nodes,
    "model_name": 10,
    "hs": 1000,
    "ls": 100,
    "m": 4,
    "bw_dist": 1,
    "bw_min": 1000.0,
    "bw_max": 10000.0,
    "p": 0.15,
    "q": 0.2,
}

config_file = topo_dir / "brite_config.conf"
brite_gen.generate(str(config_file), **config_params)
print(f"BRITE config saved to: {config_file}")

# Step 2: Run BRITE to generate topology
print("\n2. Running BRITE...")
brite_stem = topo_dir / "topology"
brite_output = run_brite(Path(config_file), Path(brite_stem), brite_path=BRITE_PATH)
print(f"BRITE topology saved to: {brite_output}")

# Step 3: Convert to SCION topology (includes random peering + step PNGs)
print("\n3. Converting to SCION topology...")
converter = BRITE2SCIONConverter()
scion_topo = converter.convert_brite_file(brite_output, plot_dir=topo_dir)

G = scion_topo["graph"]

topology_file = topo_dir / "scion_topology.pkl"
with open(topology_file, "wb") as f:
    pickle.dump(scion_topo, f)
print(f"SCION topology saved to: {topology_file}")

json_data = {
    "isds": scion_topo["isds"],
    "core_ases": list(scion_topo["core_ases"]),
    "graph": nx.node_link_data(G),
}
json_file = topo_dir / "scion_topology.json"
with open(json_file, "w") as f:
    json.dump(json_data, f, indent=2, default=str)
print(f"SCION topology JSON saved to: {json_file}")

# Print statistics
print("\n4. Topology Statistics:")
print(f"   - Total ASes: {G.number_of_nodes()}")
print(f"   - Total links: {G.number_of_edges()}")
print(f"   - ISDs: {len(scion_topo['isds'])}")
print(f"   - Core ASes: {len(scion_topo['core_ases'])}")

link_types: dict[str, int] = {}
for _, _, data in G.edges(data=True):
    link_type = str(data.get("type", "UNKNOWN"))
    link_types[link_type] = link_types.get(link_type, 0) + 1
print("   - Link types:")
for lt, count in sorted(link_types.items()):
    print(f"     - {lt}: {count}")

if nx.is_connected(G.to_undirected()):
    print("   - Graph is connected: Yes")
else:
    print("   - Graph is connected: No")
    components = list(nx.connected_components(G.to_undirected()))
    print(f"   - Number of components: {len(components)}")

avg_degree = sum(dict(G.degree()).values()) / len(G)
print(f"   - Average degree: {avg_degree:.2f}")

print(f"\nTopology generation complete! Artifacts in {topo_dir}/")
