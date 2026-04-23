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
