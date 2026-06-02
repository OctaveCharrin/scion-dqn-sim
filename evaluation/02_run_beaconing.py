#!/usr/bin/env python3
"""Run SCION beaconing and populate the multi-pair path store (pipeline step 02)."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.beacon_pipeline import run_beaconing

if __name__ == "__main__":
    run_path = Path(resolve_run_dir())
    run_beaconing(run_path)
    print("\nBeaconing complete.")
