"""Run-directory resolution and subprocess execution for the evaluation pipeline."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

PIPELINE_STEP_SCRIPTS: tuple[str, ...] = (
    "01_generate_topology.py",
    "02_run_beaconing.py",
    "03_simulate_traffic.py",
    "04_train_dqn.py",
    "04_train_simple_dqn.py",
    "04_train_scoring_dqn.py",
    "04_train_scoring_enhanced_dqn.py",
    "04_train_conditional_dqn.py",
    "05_evaluate_methods.py",
    "eval_multi_reward_comparison.py",
    "06_generate_figures.py",
)


def resolve_run_dir(
    argv: Optional[Sequence[str]] = None,
    *,
    cwd: Optional[Path] = None,
    must_exist: bool = True,
) -> str:
    """Resolve a ``run_*`` directory from argv or the most recent run in ``cwd``."""
    argv = sys.argv if argv is None else argv
    if len(argv) > 1 and argv[1]:
        run_dir = argv[1]
    else:
        search_dir = Path(cwd) if cwd else Path.cwd()
        dirs = sorted(
            d.name
            for d in search_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        )
        if not dirs:
            if must_exist:
                raise FileNotFoundError(
                    f"No run_* directory found in {search_dir}. Pass one as argv[1]."
                )
            return ""
        run_dir = dirs[-1]

    print(f"Using run directory: {run_dir}")
    return run_dir


def pipeline_steps_from(from_step: int) -> List[str]:
    """Return pipeline scripts from ``from_step`` (1-based) through step 06."""
    if from_step < 1 or from_step > len(PIPELINE_STEP_SCRIPTS):
        raise ValueError(
            f"from_step must be between 1 and {len(PIPELINE_STEP_SCRIPTS)}, got {from_step}"
        )
    return list(PIPELINE_STEP_SCRIPTS[from_step - 1 :])


def resolve_existing_run_dir(
    run_dir: Optional[str] = None,
    *,
    cwd: Optional[Path] = None,
) -> str:
    """Resolve an existing ``run_*`` directory (never create a new one)."""
    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / path
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    search_dir = Path(cwd) if cwd else Path.cwd()
    dirs = sorted(
        d.name for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    )
    if not dirs:
        raise FileNotFoundError(
            f"No run_* directory found in {search_dir}. "
            "Pass --run-dir PATH to an existing run with steps 01–03 complete."
        )
    return dirs[-1]


def run_script(
    script_name: str,
    run_dir: Optional[str] = None,
    *,
    cwd: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
) -> str:
    """Execute a numbered pipeline script and print its output."""
    banner = "=" * 60
    print(f"\n{banner}\nRunning {script_name}...\n{banner}")

    cmd = [sys.executable, script_name]
    if run_dir:
        cmd.append(run_dir)
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR in {script_name}:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)
    print(f"Completed in {elapsed:.1f} seconds")
    return result.stdout
