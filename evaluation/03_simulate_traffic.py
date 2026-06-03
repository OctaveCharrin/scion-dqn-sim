#!/usr/bin/env python3
"""Simulate 28 days of traffic on the SCION topology (pipeline step 03).

Uses calibrated sparse demand (see ``TrafficSimConfig``). After the run, inspect
congestion with ``evaluation/inspect_traffic.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.link_traffic_sim import simulate_link_traffic
from src.simulation.traffic_config import TrafficSimConfig

if __name__ == "__main__":
    run_path = Path(resolve_run_dir())
    simulate_link_traffic(run_path, TrafficSimConfig.from_env())
    print("\nTip: uv run python inspect_traffic.py", run_path.name)
