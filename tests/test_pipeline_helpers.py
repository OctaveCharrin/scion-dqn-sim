"""Tests for pipeline orchestration helpers (``src.pipeline``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import figures, run_dirs
from src.simulation.run_context import TOPOLOGY_SUBDIR_NAME, topology_dir, validate_pre_training_artifacts


def test_resolve_run_dir_argv(tmp_path: Path, capsys):
    run_dir = run_dirs.resolve_run_dir(["script.py", "my_run"], cwd=tmp_path)
    assert run_dir == "my_run"


def test_resolve_run_dir_latest(tmp_path: Path, capsys):
    (tmp_path / "run_20260101_000000").mkdir()
    (tmp_path / "run_20260102_000000").mkdir()
    (tmp_path / "not_a_run").mkdir()

    run_dir = run_dirs.resolve_run_dir(["script.py"], cwd=tmp_path)
    assert run_dir == "run_20260102_000000"


def test_resolve_run_dir_errors_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_dirs.resolve_run_dir(["script.py"], cwd=tmp_path)


def test_topology_dir(tmp_path: Path):
    run = tmp_path / "run_20260101_120000"
    run.mkdir()
    td = topology_dir(run)
    assert td == run / TOPOLOGY_SUBDIR_NAME
    assert td.name == "topology"


def test_method_name_and_color_defaults():
    assert figures.display_name("dqn") == "DQN (Enhanced)"
    assert figures.display_name("unknown-method") == "unknown-method"
    assert figures.color_for("dqn").startswith("#")
    assert figures.color_for("unknown") == "#333333"


def test_figure_widths_are_positive():
    assert figures.COLUMN_WIDTH > 0
    assert figures.FULL_WIDTH >= figures.COLUMN_WIDTH


def test_pipeline_steps_from():
    full = run_dirs.pipeline_steps_from(1)
    assert full[0] == "01_generate_topology.py"
    assert full[-1] == "06_generate_figures.py"
    from_four = run_dirs.pipeline_steps_from(4)
    assert from_four == [
        "04_train_all_models.py",
        "05_evaluate_methods.py",
        "eval_multi_reward_comparison.py",
        "06_generate_figures.py",
    ]
    with pytest.raises(ValueError):
        run_dirs.pipeline_steps_from(0)


def test_resolve_existing_run_dir_never_creates(tmp_path: Path):
    (tmp_path / "run_20260101_000000").mkdir()
    assert run_dirs.resolve_existing_run_dir(cwd=tmp_path) == "run_20260101_000000"
    assert (
        run_dirs.resolve_existing_run_dir("run_20260101_000000", cwd=tmp_path)
        == "run_20260101_000000"
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run_dirs.resolve_existing_run_dir(cwd=empty)


def test_validate_pre_training_artifacts(tmp_path: Path):
    run = tmp_path / "run_test"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="topology"):
        validate_pre_training_artifacts(run)

    topo = run / TOPOLOGY_SUBDIR_NAME
    topo.mkdir()
    (topo / "scion_topology.json").write_text("{}")
    for name in (
        "path_store.json",
        "selected_pair.json",
        "traffic_flows.pkl",
        "link_states.pkl",
    ):
        (run / name).write_bytes(b"x")
    validate_pre_training_artifacts(run)
