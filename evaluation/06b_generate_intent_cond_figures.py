#!/usr/bin/env python3
"""Render intent conditioning figures (6.1, 6.2, 6.3) from the intent conditioning CSV artifacts.

Reads the CSVs written by the conditioning eval scripts and writes PNGs into
``<artifact-dir>/figures/``. Point ``--artifact-dir`` at a ``intent_cond_*`` dir
(default: the newest ``intent_cond_*`` under the resolved run dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.intent_cond_figures import generate_all_figures
from src.pipeline.run_dirs import resolve_run_dir


def _latest_artifact_dir(run_path: Path) -> Path:
    candidates = sorted(
        d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("intent_cond_")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No intent_cond_* artifact dir under {run_path}. Run the eval scripts first."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="intent_cond_* dir holding the CSVs (default: newest under run dir).",
    )
    parser.add_argument(
        "--metric",
        choices=["goodput", "reward", "both"],
        default="goodput",
        help="QoE metric for Figs 6.2/6.3 (default: goodput).",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    artifact_dir = args.artifact_dir or _latest_artifact_dir(run_path)

    print(f"Generating intent conditioning figures from {artifact_dir} (metric={args.metric})")
    outputs = generate_all_figures(artifact_dir, metric=args.metric)
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    if not outputs:
        print("  (no CSVs found — nothing rendered)")


if __name__ == "__main__":
    main()
