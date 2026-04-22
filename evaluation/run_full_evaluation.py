#!/usr/bin/env python3
"""Run the complete SCION DQN evaluation pipeline.

Orchestrates the 6 numbered steps into a single run directory:
    01_generate_topology.py
    02_run_beaconing.py
    03_simulate_traffic.py
    04_train_dqn.py
    05_evaluate_methods.py
    06_generate_figures.py
"""

import argparse
import os
from datetime import datetime

from _common import run_script


PIPELINE_STEPS = [
    "01_generate_topology.py",
    "02_run_beaconing.py",
    "03_simulate_traffic.py",
    "04_train_dqn.py",
    "05_evaluate_methods.py",
    "06_generate_figures.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing run directory to reuse (default: create a new timestamped run_*)",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Using run directory: {run_dir}")

    for step in PIPELINE_STEPS:
        run_script(step, run_dir)

    banner = "=" * 60
    print(f"\n{banner}\nEVALUATION COMPLETE!\n{banner}")
    print(f"\nAll results saved in: {run_dir}/")
    print("\nKey outputs:")
    for name, desc in [
        ("scion_topology.json", "Network topology"),
        ("selected_pair.json", "Source-destination pair"),
        ("dqn_model.pth", "Trained DQN model"),
        ("evaluation_results.json", "Performance comparison"),
        ("figure1_probe_overhead.pdf", "Probe overhead comparison"),
        ("figure2_path_reward.pdf", "Path reward distribution"),
        ("figure3_probe_breakdown.pdf", "Probe type breakdown"),
    ]:
        print(f"  - {run_dir}/{name}: {desc}")


if __name__ == "__main__":
    main()
