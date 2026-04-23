#!/usr/bin/env python3
"""
Plot topology from an evaluation run (``scion_topology.json`` or pickle).

Modes
-----
**full** — dashboard: large geographic map + degree histogram + ISD composition
stacked bars + link-type pie chart; optionally extra PNGs (ISD map, core-only
graph, connectivity matrix).

**simple** — single geographic figure with AS / link legends (lighter file).

Examples
--------
  cd evaluation
  uv run python visualize_topology.py
  uv run python visualize_topology.py run_20260422_120000
  uv run python visualize_topology.py --mode simple
  uv run python visualize_topology.py -t path/to/scion_topology.json -o out.png --mode full
  uv run python visualize_topology.py --report --no-extras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401  — prepends repo root to sys.path

from src.visualization.topology_visualizer import (
    TopologyVisualizer,
    generate_topology_stats,
    load_topology_tables,
    render_scion_topology_png,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Evaluation run directory (e.g. run_20260101_120000). Ignored if --topology is set.",
    )
    parser.add_argument(
        "--topology",
        "-t",
        type=Path,
        default=None,
        help="Path to scion_topology.json or a topology pickle.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "simple"),
        default="full",
        help="full = dashboard + optional extras; simple = one geographic PNG.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path. For 'full', use a .png file path or a directory "
        "(dashboard written as topology_dashboard.png inside).",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="With --mode full, skip isd_map / core_network / connectivity_matrix PNGs.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write topology_stats.txt next to the main output.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide AS ID labels on the geographic map(s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for simple mode and dashboard (default: 200; dashboard save uses this).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="With --mode simple, open an interactive window after saving.",
    )
    args = parser.parse_args()

    if args.topology is not None:
        topo_path = args.topology.resolve()
        base_dir = topo_path.parent
    else:
        run_dir = args.run_dir
        if not run_dir:
            from _common import resolve_run_dir

            run_dir = resolve_run_dir()
        base_dir = Path(run_dir).resolve()
        topo_json = base_dir / "scion_topology.json"
        topo_pkl = base_dir / "scion_topology.pkl"
        if topo_json.is_file():
            topo_path = topo_json
        elif topo_pkl.is_file():
            topo_path = topo_pkl
        else:
            raise SystemExit(
                f"No scion_topology.json or scion_topology.pkl under {base_dir}"
            )

    if not topo_path.is_file():
        raise SystemExit(f"Topology file not found: {topo_path}")

    if args.mode == "simple":
        if args.output:
            out = args.output.resolve()
            if out.suffix.lower() != ".png":
                out.mkdir(parents=True, exist_ok=True)
                out = out / "topology_geographic.png"
        else:
            out = base_dir / "topology_geographic.png"
        render_scion_topology_png(
            topo_path,
            out,
            show_labels=not args.no_labels,
            dpi=args.dpi,
            show_interactive=args.show,
        )
        print(f"Saved simple topology map: {out}")
        if args.report:
            node_df, edge_df, _ = load_topology_tables(topo_path)
            stats_path = out.parent / "topology_stats.txt"
            stats_path.write_text(
                generate_topology_stats({"nodes": node_df, "edges": edge_df}),
                encoding="utf-8",
            )
            print(f"Wrote statistics: {stats_path}")
        return

    # --- full dashboard ---
    if args.output:
        out = args.output.resolve()
        if out.suffix.lower() != ".png":
            out.mkdir(parents=True, exist_ok=True)
            dash_path = out / "topology_dashboard.png"
            stats_dir = out
        else:
            dash_path = out
            stats_dir = out.parent
    else:
        dash_path = base_dir / "topology_dashboard.png"
        stats_dir = base_dir

    vis = TopologyVisualizer(figsize=(16, 12))
    vis.visualize_topology(
        topo_path,
        dash_path,
        show_labels=not args.no_labels,
        show_grid=True,
        write_extras=not args.no_extras,
        dpi=args.dpi,
    )
    print(f"Saved full topology dashboard: {dash_path}")
    if not args.no_extras:
        print(f"  (extras in {dash_path.parent}/: isd_map.png, core_network.png, connectivity_matrix.png)")

    if args.report:
        node_df, edge_df, _ = load_topology_tables(topo_path)
        stats_path = stats_dir / "topology_stats.txt"
        stats_path.write_text(
            generate_topology_stats({"nodes": node_df, "edges": edge_df}),
            encoding="utf-8",
        )
        print(f"Wrote statistics: {stats_path}")


if __name__ == "__main__":
    main()
