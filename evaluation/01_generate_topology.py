#!/usr/bin/env python3
"""
Generate dense 50-AS SCION topology using BRITE
"""

import os
import sys
import json
import pickle
import subprocess
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.topology.brite_cfg_gen import BRITEConfigGenerator
from src.topology.brite2scion_converter import BRITE2SCIONConverter
# from src.visualization.topology_visualizer import plot_scion_topology

def run_brite(config_path: str, output_stem: str) -> str:
    """
    Run BRITE (Main.Brite). ``output_stem`` must be a path *without* the ``.brite`` suffix;
    BRITE writes ``<output_stem>.brite``. Returns the path to that file.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    brite_path = os.path.join(repo_root, "external", "brite")
    jar_path = os.path.join(brite_path, "Java", "Brite.jar")
    seed_path = os.path.join(brite_path, "Java", "seed_file")

    if not os.path.exists(jar_path):
        raise RuntimeError(f"BRITE jar not found at {jar_path}. Run ./setup_brite.sh")
    if not os.path.exists(seed_path):
        raise RuntimeError(f"BRITE seed file missing at {seed_path}")

    config_path = str(Path(config_path).resolve())
    output_stem = str(Path(output_stem).resolve())

    cmd = [
        "java",
        "-jar",
        jar_path,
        config_path,
        output_stem,
        seed_path,
    ]
    result = subprocess.run(
        cmd,
        cwd=brite_path,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "[ERROR]" in combined:
        raise RuntimeError(
            f"BRITE generation failed (exit {result.returncode}):\n{combined}"
        )

    brite_out = output_stem + ".brite"
    if not os.path.isfile(brite_out):
        raise RuntimeError(f"BRITE did not create expected file: {brite_out}\n{combined}")
    return brite_out

if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    run_dir = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(run_dir, exist_ok=True)
    print(f"Using run directory: {run_dir}")

print(f"Creating dense SCION topology in {run_dir}")

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

config_file = os.path.join(run_dir, "brite_config.conf")
brite_gen.generate(config_file, **config_params)
print(f"BRITE config saved to: {config_file}")

# Step 2: Run BRITE to generate topology
print("\n2. Running BRITE...")
brite_stem = os.path.join(run_dir, "topology")
brite_output = run_brite(config_file, brite_stem)
print(f"BRITE topology saved to: {brite_output}")

# Step 3: Convert to SCION topology
print("\n3. Converting to SCION topology...")
converter = BRITE2SCIONConverter()
scion_topo = converter.convert_brite_file(brite_output)

# Get the graph
G = scion_topo['graph']
nodes = list(G.nodes())
np.random.seed(42)

# Step 4: Enhance connectivity with additional peering links
print("\n4. Adding peering links for dense connectivity...")
num_peering_to_add = min(75, max(2, len(nodes) * len(nodes) // 4))
added = 0
interface_id = 1000  # Start peering interfaces at 1000

for _ in range(2000):  # Try many times
    # Use Python int endpoints: numpy types + json.dump(..., default=str) would
    # stringify node ids on edges, and node_link_graph would treat "2" and 2 as
    # different ASes (inflated AS count after reload).
    src, dst = map(int, np.random.choice(nodes, 2, replace=False))
    
    # Skip if already connected
    if G.has_edge(src, dst) or G.has_edge(dst, src):
        continue
    
    # Skip if in same ISD (prefer inter-ISD peering)
    if G.nodes[src].get('isd') == G.nodes[dst].get('isd'):
        if np.random.random() > 0.3:  # Still allow some intra-ISD peering
            continue
    
    # Add bidirectional peering link
    G.add_edge(src, dst, 
               src_if=interface_id,
               dst_if=interface_id + 1,
               type='PEER',
               bandwidth=np.random.uniform(5000, 10000),
               latency=np.random.uniform(5, 25))
    
    G.add_edge(dst, src,
               src_if=interface_id + 1,
               dst_if=interface_id,
               type='PEER',
               bandwidth=np.random.uniform(5000, 10000),
               latency=np.random.uniform(5, 25))
    
    interface_id += 2
    added += 1
    if added >= num_peering_to_add:
        break

print(f"Added {added} additional peering links")

# Save enhanced topology
topology_file = os.path.join(run_dir, "scion_topology.pkl")
with open(topology_file, 'wb') as f:
    pickle.dump(scion_topo, f)
print(f"SCION topology saved to: {topology_file}")

# Also save as JSON for inspection
json_data = {
    'isds': scion_topo['isds'],
    'core_ases': list(scion_topo['core_ases']),
    'graph': nx.node_link_data(G)
}
json_file = os.path.join(run_dir, "scion_topology.json")
with open(json_file, 'w') as f:
    json.dump(json_data, f, indent=2, default=str)

# Print statistics
print("\n5. Topology Statistics:")
print(f"   - Total ASes: {G.number_of_nodes()}")
print(f"   - Total links: {G.number_of_edges()}")
print(f"   - ISDs: {len(scion_topo['isds'])}")
print(f"   - Core ASes: {len(scion_topo['core_ases'])}")

# Count link types
link_types = {}
for _, _, data in G.edges(data=True):
    link_type = data.get('type', 'UNKNOWN')
    link_types[link_type] = link_types.get(link_type, 0) + 1
print("   - Link types:")
for lt, count in sorted(link_types.items()):
    print(f"     - {lt}: {count}")

# Check connectivity
if nx.is_connected(G.to_undirected()):
    print("   - Graph is connected: Yes")
else:
    print("   - Graph is connected: No")
    components = list(nx.connected_components(G.to_undirected()))
    print(f"   - Number of components: {len(components)}")

# Calculate average degree
avg_degree = sum(dict(G.degree()).values()) / len(G)
print(f"   - Average degree: {avg_degree:.2f}")

# Visualize topology
print("\n6. Generating visualization...")
# vis_file = os.path.join(run_dir, "topology_visualization.png")
# plot_scion_topology(scion_topo, output_path=vis_file, title="50-AS Dense SCION Topology")
# print(f"Visualization saved to: {vis_file}")
print("Visualization skipped (module not available)")

print(f"\nTopology generation complete! Files saved in {run_dir}/")