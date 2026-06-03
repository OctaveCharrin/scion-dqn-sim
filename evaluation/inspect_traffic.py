#!/usr/bin/env python3
"""Inspect congestion and traffic calibration for a pipeline run (step 03 output)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.traffic_inspect import analyze_traffic_run, format_traffic_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect traffic / congestion metrics.")
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full report JSON.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    report = analyze_traffic_run(run_path)
    print(format_traffic_report(report))

    out = args.json_out or (run_path / "traffic_inspection.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
