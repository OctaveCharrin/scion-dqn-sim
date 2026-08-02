#!/usr/bin/env python3
"""Render the six figures with the across-seed spread and, optionally,
install them into the thesis figure directory under their existing names.

The single-run versions are produced by ``06b_generate_intent_cond_figures.py``;
these read the aggregates from ``analyze_seed_results.py`` instead. Filenames are
kept identical (``p1eval_*.png``) so the thesis picks them up with no edit to any
``\\includegraphics``.

Usage:
    uv run python plot_seed_figures.py <run_dir> [--copy-to ~/thesis-report/figures]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.pipeline.five_seed_figures import generate_seed_figures
from src.pipeline.run_dirs import resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help="Directory written by analyze_seed_results.py "
        "(default: <run>/seeds/aggregate).",
    )
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Also copy each rendered figure into this directory (the thesis "
        "figures dir), overwriting the single-run version.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    agg = args.aggregate_dir or (run_path / "seeds" / "aggregate")
    if not (agg / "claims.json").is_file():
        raise SystemExit(f"No aggregates in {agg}. Run analyze_seed_results.py first.")

    figs = generate_seed_figures(agg, args.fig_dir)
    print(f"Seeded figures from {agg}")
    for name, path in figs.items():
        print(f"  {name}: {path}")

    if args.copy_to:
        dest_dir = args.copy_to.expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nInstalling into {dest_dir}")
        for name, path in figs.items():
            dest = dest_dir / name
            shutil.copyfile(path, dest)
            print(f"  {dest}")


if __name__ == "__main__":
    main()
