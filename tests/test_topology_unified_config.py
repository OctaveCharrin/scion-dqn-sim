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
    assert cfg.get("generator") in ("brite", None)
    assert "brite" in cfg
    assert "output" in cfg


