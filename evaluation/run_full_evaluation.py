#!/usr/bin/env python3
"""Run the complete SCION DQN evaluation pipeline.

Orchestrates the numbered steps into a single run directory:
    01_generate_topology.py
    02_run_beaconing.py
    03_simulate_traffic.py
    04_train_dqn.py
    04_train_simple_dqn.py
    04_train_scoring_dqn.py
    04_train_scoring_enhanced_dqn.py
    05_evaluate_methods.py
    06_generate_figures.py

Resume training on an existing run (skip topology / beaconing / traffic):

    uv run python run_full_evaluation.py --from-step 4 --run-dir run_YYYYMMDD_HHMMSS

If ``--run-dir`` is omitted with ``--from-step 4``, the latest ``run_*`` in the
current directory is used (nothing new is created).
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from _common import (
    TOPOLOGY_SUBDIR_NAME,
    pipeline_steps_from,
    resolve_existing_run_dir,
    run_script,
    validate_pre_training_artifacts,
)

_EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory (default: new timestamped run_*; with --from-step 4: latest run_* if omitted)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        metavar="N",
        help="First pipeline step to run, 1–9 (default: 1 = full pipeline). "
        "Use 4 to reuse existing topology, path store, and traffic from steps 01–03.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to an existing BRITE config file to use instead of generating one.",
    )
    parser.add_argument(
        "--topology-config",
        "-C",
        type=Path,
        default=None,
        help="YAML file overriding defaults (merged on top of topology_defaults.yaml).",
    )
    args = parser.parse_args()

    try:
        steps = pipeline_steps_from(args.from_step)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    resume_training = args.from_step >= 4

    if resume_training:
        try:
            run_dir = resolve_existing_run_dir(args.run_dir, cwd=_EVAL_DIR)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        try:
            validate_pre_training_artifacts(run_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        skipped = pipeline_steps_from(1)[: args.from_step - 1]
        print(f"Using run directory: {run_dir}")
        print(f"Resuming from step {args.from_step} (skipping: {', '.join(skipped)})")
    else:
        if args.from_step > 1:
            print(
                f"ERROR: --from-step {args.from_step} requires an existing run; "
                "use --from-step 4 or run the full pipeline from step 1.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.run_dir:
            run_dir = args.run_dir
        else:
            run_dir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"Using run directory: {run_dir}")

    for step in steps:
        extra_args = []
        if step == "01_generate_topology.py":
            if args.config_path:
                extra_args.extend(["--config", args.config_path])
            if args.topology_config:
                extra_args.extend(["--topology-config", args.topology_config])
        run_script(step, run_dir, extra_args=extra_args, cwd=_EVAL_DIR)

    banner = "=" * 60
    print(f"\n{banner}\nEVALUATION COMPLETE!\n{banner}")
    print(f"\nAll results saved in: {run_dir}/")
    print("\nKey outputs:")
    for name, desc in [
        (f"{TOPOLOGY_SUBDIR_NAME}/scion_topology.json", "Network topology"),
        ("selected_pair.json", "Source-destination pair"),
        ("dqn_model.pth", "Trained DQN model (enhanced)"),
        ("dqn_scoring_simple_model.pth", "Trained simple path-scoring DQN"),
        ("dqn_scoring_enhanced_model.pth", "Trained enhanced path-scoring DQN"),
        ("evaluation_results.json", "Performance comparison"),
        ("figure1_probe_overhead.png", "Probe overhead comparison"),
        ("figure2_path_reward.png", "Path reward distribution"),
        ("figure3_probe_breakdown.png", "Probe type breakdown"),
    ]:
        print(f"  - {run_dir}/{name}: {desc}")


if __name__ == "__main__":
    main()
