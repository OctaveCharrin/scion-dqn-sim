#!/usr/bin/env python3
"""Simulate 28 days of traffic on the SCION topology (pipeline step 03)."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.link_traffic_sim import simulate_link_traffic

if __name__ == "__main__":
    run_path = Path(resolve_run_dir())
    simulate_link_traffic(run_path)
