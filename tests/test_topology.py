"""Tests for BRITE config generation and topology helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.topology.brite_cfg_gen import (
    AS_BARABASI2,
    BRITEConfigGenerator,
)


def test_brite_config_generation_writes_expected_fields():
    gen = BRITEConfigGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.conf"
        gen.generate(out, n_nodes=50, seed=42)
        assert out.exists()
        content = out.read_text()
        # BRITE config uses capital ``N`` for node count.
        assert "N = 50" in content
        assert "BeginModel" in content
        assert "BeginOutput" in content
        assert "BRITE = 1" in content


def test_brite_config_includes_pq_for_barabasi2():
    gen = BRITEConfigGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "ba2.conf"
        gen.generate(
            out,
            model_name=AS_BARABASI2,
            n_nodes=25,
            m=3,
            p=0.4,
            q=0.15,
        )
        content = out.read_text()
        assert "p = 0.4" in content
        assert "q = 0.15" in content


def test_brite_config_legacy_num_as_alias():
    gen = BRITEConfigGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "legacy.conf"
        gen.generate(out, num_as=12)
        content = out.read_text()
        assert "N = 12" in content


def test_run_brite_raises_when_jar_missing(tmp_path: Path):
    from src.topology.brite_cfg_gen import run_brite

    fake_brite = tmp_path / "fake_brite"
    (fake_brite / "Java").mkdir(parents=True)
    # No jar / seed file present.
    cfg = tmp_path / "c.conf"
    cfg.write_text("BriteConfig\n")
    with pytest.raises(FileNotFoundError):
        run_brite(cfg, tmp_path / "out", brite_path=fake_brite)
