#!/usr/bin/env python3
"""Generate SCION evaluation topology (BRITE + converter).

All artifacts written under ``<run_dir>/topology/``. Configuration is loaded
from ``topology_defaults.yaml`` (see ``src.topology.eval_config``).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import networkx as nx
import yaml

from src.simulation.run_context import topology_dir
from src.topology.eval_config import (
    load_unified_topology_config,
    nested_get,
    run_brite_topology_generation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SCION topology from unified YAML (BRITE)."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run directory (created if missing when omitted).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="YAML file overriding defaults (merged on top of topology_defaults.yaml).",
    )
    args = parser.parse_args()

    cfg = load_unified_topology_config(args.config)

    if args.run_dir:
        run_dir = args.run_dir
        print(f"Using run directory: {run_dir}")
    else:
        run_dir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"Using run directory: {run_dir}")

    topo_dir = topology_dir(run_dir)
    topo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Topology artifacts directory: {topo_dir}")

    if nested_get(cfg, "output", "dump_resolved_config", default=True):
        dump_path = topo_dir / "topology_config_resolved.yaml"
        with open(dump_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote resolved config: {dump_path}")

    save_png = bool(nested_get(cfg, "output", "save_step_pngs", default=True))
    generator = (
        str(nested_get(cfg, "generator", default="brite")).lower().replace("-", "_")
    )

    if generator == "brite":
        print("\n=== Topology generator: brite ===\n")
        br_rel = nested_get(cfg, "brite", "brite_path", default="external/brite")
        br_path = Path(str(br_rel))
        if not br_path.is_absolute():
            br_path = (REPO_ROOT / br_path).resolve()
        scion_topo = run_brite_topology_generation(
            cfg, topo_dir, br_path, save_png=save_png
        )
    else:
        raise SystemExit(
            f"Unknown generator: {generator!r} (only 'brite' is supported)"
        )

    G = scion_topo["graph"]

    json_data = {
        "isds": scion_topo["isds"],
        "core_ases": list(scion_topo["core_ases"]),
        "graph": nx.node_link_data(G),
    }
    json_file = topo_dir / "scion_topology.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nSCION topology JSON saved to: {json_file}")

    print("\nTopology Statistics:")
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

    avg_degree = sum(dict(G.degree()).values()) / max(len(G), 1)
    print(f"   - Average degree: {avg_degree:.2f}")

    print(f"\nTopology generation complete! Artifacts in {topo_dir}/")


if __name__ == "__main__":
    main()
