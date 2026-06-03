"""CLI for plotting SCION topologies from evaluation runs or standalone files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.run_context import topology_dir
from src.visualization.topology_visualizer import (
    TopologyVisualizer,
    generate_topology_stats,
    load_topology_tables,
    render_scion_topology_png,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot topology from an evaluation run (scion_topology.json or pickle) "
            "or a direct file path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Evaluation run directory. Ignored if --topology is set.",
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
        help="Output path (.png file or directory for full mode).",
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
        help="DPI for simple mode and dashboard (default: 200).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="With --mode simple, open an interactive window after saving.",
    )
    args = parser.parse_args(argv)

    if args.topology is not None:
        topo_path = args.topology.resolve()
        base_dir = topo_path.parent
    else:
        run_dir = args.run_dir or resolve_run_dir(must_exist=True)
        base_dir = Path(run_dir).resolve()
        tdir = topology_dir(base_dir)
        topo_json = tdir / "scion_topology.json"
        if topo_json.is_file():
            topo_path = topo_json
        else:
            raise SystemExit(
                f"No scion_topology.json under {tdir}. Run 01_generate_topology.py first."
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
        print(
            f"  (extras in {dash_path.parent}/: "
            "isd_map.png, core_network.png, connectivity_matrix.png)"
        )

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
