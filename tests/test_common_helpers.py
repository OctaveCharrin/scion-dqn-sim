"""Tests for the shared evaluation helpers (``evaluation/_common.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def common_module():
    """Import ``evaluation/_common.py`` without needing a package install."""
    path = Path(__file__).resolve().parent.parent / "evaluation" / "_common.py"
    spec = importlib.util.spec_from_file_location("evaluation_common", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluation_common"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_run_dir_argv(tmp_path: Path, common_module, capsys):
    run_dir = common_module.resolve_run_dir(["script.py", "my_run"], cwd=tmp_path)
    assert run_dir == "my_run"


def test_resolve_run_dir_latest(tmp_path: Path, common_module, capsys):
    (tmp_path / "run_20260101_000000").mkdir()
    (tmp_path / "run_20260102_000000").mkdir()
    (tmp_path / "not_a_run").mkdir()

    run_dir = common_module.resolve_run_dir(["script.py"], cwd=tmp_path)
    assert run_dir == "run_20260102_000000"


def test_resolve_run_dir_errors_when_missing(tmp_path: Path, common_module):
    with pytest.raises(FileNotFoundError):
        common_module.resolve_run_dir(["script.py"], cwd=tmp_path)


def test_topology_dir(common_module, tmp_path: Path):
    run = tmp_path / "run_20260101_120000"
    run.mkdir()
    td = common_module.topology_dir(run)
    assert td == run / common_module.TOPOLOGY_SUBDIR_NAME
    assert td.name == "topology"


def test_method_name_and_color_defaults(common_module):
    assert common_module.display_name("dqn") == "DQN (Ours)"
    assert common_module.display_name("unknown-method") == "unknown-method"
    assert common_module.color_for("dqn").startswith("#")
    assert common_module.color_for("unknown") == "#333333"


def test_figure_widths_are_positive(common_module):
    assert common_module.COLUMN_WIDTH > 0
    assert common_module.FULL_WIDTH >= common_module.COLUMN_WIDTH


def test_pipeline_steps_from(common_module):
    full = common_module.pipeline_steps_from(1)
    assert full[0] == "01_generate_topology.py"
    assert full[-1] == "06_generate_figures.py"
    from_four = common_module.pipeline_steps_from(4)
    assert from_four == [
        "04_train_dqn.py",
        "04_train_simple_dqn.py",
        "04_train_scoring_dqn.py",
        "04_train_scoring_enhanced_dqn.py",
        "05_evaluate_methods.py",
        "06_generate_figures.py",
    ]
    with pytest.raises(ValueError):
        common_module.pipeline_steps_from(0)


def test_resolve_existing_run_dir_never_creates(tmp_path: Path, common_module):
    (tmp_path / "run_20260101_000000").mkdir()
    assert common_module.resolve_existing_run_dir(cwd=tmp_path) == "run_20260101_000000"
    assert common_module.resolve_existing_run_dir("run_20260101_000000", cwd=tmp_path) == "run_20260101_000000"
    with pytest.raises(FileNotFoundError):
        common_module.resolve_existing_run_dir(cwd=tmp_path)


def test_validate_pre_training_artifacts(common_module, tmp_path: Path):
    run = tmp_path / "run_test"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="topology"):
        common_module.validate_pre_training_artifacts(run)

    topo = run / common_module.TOPOLOGY_SUBDIR_NAME
    topo.mkdir()
    (topo / "scion_topology.json").write_text("{}")
    for name in ("path_store.json", "selected_pair.json", "traffic_flows.pkl", "link_states.pkl"):
        (run / name).write_bytes(b"x")
    common_module.validate_pre_training_artifacts(run)
