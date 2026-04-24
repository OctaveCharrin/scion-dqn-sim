"""Sanity checks for evaluation topology YAML defaults."""

from __future__ import annotations

from pathlib import Path

import yaml


def _defaults_path() -> Path:
    return Path(__file__).resolve().parent.parent / "evaluation" / "topology_defaults.yaml"


def test_topology_defaults_yaml_exists_and_parses() -> None:
    p = _defaults_path()
    assert p.is_file(), "Ship evaluation/topology_defaults.yaml with the repo"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert cfg.get("generator") in ("brite", "top_down", None)
    assert "brite" in cfg
    assert "top_down" in cfg
    assert "output" in cfg


def test_fixture_top_down_yaml() -> None:
    fx = Path(__file__).resolve().parent / "fixtures" / "topology_min_top_down.yaml"
    cfg = yaml.safe_load(fx.read_text(encoding="utf-8"))
    assert cfg["generator"] == "top_down"
    assert cfg["top_down"]["n_nodes"] == 35
