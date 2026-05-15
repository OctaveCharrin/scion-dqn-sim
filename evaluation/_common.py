"""Shared helpers for the numbered evaluation scripts.

Centralizes repeated utilities (run-directory resolution, pipeline step
execution, figure styling, method display metadata) so each step script stays
small and declarative.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

# Ensure the repo root is importable so scripts can ``from src.<x> import ...``
# regardless of where they are invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -----------------------------------------------------------------------------
# Topology artifact layout (under each ``run_*`` directory)
# -----------------------------------------------------------------------------

TOPOLOGY_SUBDIR_NAME = "topology"


def topology_dir(run_dir: str | Path) -> Path:
    """Return ``<run_dir>/topology`` where BRITE/SCION artifacts and step plots live."""
    return Path(run_dir) / TOPOLOGY_SUBDIR_NAME


# -----------------------------------------------------------------------------
# Run directory
# -----------------------------------------------------------------------------


def resolve_run_dir(
    argv: Optional[Sequence[str]] = None,
    *,
    cwd: Optional[Path] = None,
    must_exist: bool = True,
) -> str:
    """Resolve a ``run_*`` directory from argv or the most recent run in ``cwd``.

    Parameters
    ----------
    argv: command-line arguments (defaults to ``sys.argv``). If ``argv[1]``
        is provided it is used as the run directory.
    cwd: directory to scan for ``run_*`` folders (defaults to the current
        working directory).
    must_exist: when True (default) raise ``FileNotFoundError`` if no run
        directory is found.
    """
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


# -----------------------------------------------------------------------------
# Pipeline orchestration
# -----------------------------------------------------------------------------

PIPELINE_STEP_SCRIPTS: tuple[str, ...] = (
    "01_generate_topology.py",
    "02_run_beaconing.py",
    "03_simulate_traffic.py",
    "04_train_dqn.py",
    "04_train_simple_dqn.py",
    "04_train_scoring_dqn.py",
    "04_train_scoring_enhanced_dqn.py",
    "05_evaluate_methods.py",
    "06_generate_figures.py",
)


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
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return str(path)

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


def validate_pre_training_artifacts(run_dir: str | Path) -> None:
    """Ensure topology, beaconing, and traffic outputs exist before step 04."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    topo = topology_dir(root) / "scion_topology.json"
    if not topo.is_file():
        legacy = root / "scion_topology.json"
        if not legacy.is_file():
            raise FileNotFoundError(
                f"Missing topology JSON in {root / TOPOLOGY_SUBDIR_NAME} "
                f"or {legacy}. Run 01_generate_topology.py first."
            )

    required = [
        root / "path_store.json",
        root / "selected_pair.json",
        root / "traffic_flows.pkl",
        root / "link_states.pkl",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run directory is missing artifacts from steps 02–03:\n  "
            + "\n  ".join(missing)
            + "\nRun 02_run_beaconing.py and 03_simulate_traffic.py first."
        )


def run_script(
    script_name: str,
    run_dir: Optional[str] = None,
    *,
    cwd: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
) -> str:
    """Execute a numbered pipeline script and print its output.

    Exits with status 1 if the script fails.
    """
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


# -----------------------------------------------------------------------------
# Figure styling
# -----------------------------------------------------------------------------

# LNCS column widths (in inches).
COLUMN_WIDTH: float = 3.5
FULL_WIDTH: float = 7.0

METHOD_DISPLAY_NAMES: Mapping[str, str] = {
    "dqn": "DQN (Enhanced)",
    "simple_dqn": "DQN (Simple)",
    "scoring_simple_dqn": "DQN (Path-Scoring Simple)",
    "scoring_enhanced_dqn": "DQN (Path-Scoring Enhanced)",
    "scoring_dqn": "DQN (Path-Scoring Simple)",
    "shortest_path": "Shortest Path",
    "widest_path": "Widest Path",
    "lowest_latency": "Lowest Latency",
    "ecmp": "ECMP",
    "random": "Random",
    "scion_default": "SCION Default",
}

METHOD_COLORS: Mapping[str, str] = {
    "dqn": "#1f77b4",
    "simple_dqn": "#17becf",
    "scoring_simple_dqn": "#bcbd22",
    "scoring_enhanced_dqn": "#7f7f7f",
    "scoring_dqn": "#bcbd22",
    "shortest_path": "#ff7f0e",
    "widest_path": "#2ca02c",
    "lowest_latency": "#d62728",
    "ecmp": "#9467bd",
    "random": "#8c564b",
    "scion_default": "#e377c2",
}


def apply_lncs_style() -> None:
    """Configure matplotlib's rcParams for LNCS-style figures."""
    # Imported lazily so this module stays light for non-plotting callers.
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"]
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 9
    rcParams["figure.titlesize"] = 12


def display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


__all__ = [
    "COLUMN_WIDTH",
    "FULL_WIDTH",
    "METHOD_COLORS",
    "METHOD_DISPLAY_NAMES",
    "PIPELINE_STEP_SCRIPTS",
    "TOPOLOGY_SUBDIR_NAME",
    "apply_lncs_style",
    "color_for",
    "display_name",
    "pipeline_steps_from",
    "resolve_existing_run_dir",
    "resolve_run_dir",
    "run_script",
    "topology_dir",
    "validate_pre_training_artifacts",
]
