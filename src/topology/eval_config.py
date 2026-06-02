"""YAML topology configuration loading and BRITE generation for the evaluation pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from src.topology.brite2scion_converter import BRITE2SCIONConverter
from src.topology.brite_cfg_gen import BRITEConfigGenerator, run_brite

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_REL = REPO_ROOT / "evaluation" / "topology_defaults.yaml"


def nested_get(cfg: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_user_path(p: Any, bases: list[Path]) -> Path:
    """Resolve a config path: absolute, or first existing match under ``bases``."""
    if p is None or p == "":
        raise ValueError("path is empty")
    pp = Path(str(p)).expanduser()
    if pp.is_file():
        return pp.resolve()
    for b in bases:
        cand = (b / pp).resolve()
        if cand.is_file():
            return cand
    return pp.resolve()


def load_unified_topology_config(cli_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``topology_defaults.yaml`` then optional user YAML on top."""
    cfg: Dict[str, Any] = {}
    if DEFAULT_CONFIG_REL.is_file():
        with open(DEFAULT_CONFIG_REL, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if cli_path is not None:
        p = cli_path.expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Topology config not found: {p}")
        with open(p, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    return cfg


def _coalesce_int(*vals: Any) -> Optional[int]:
    for v in vals:
        if v is not None:
            return int(v)
    return None


def run_brite_topology_generation(
    cfg: Dict[str, Any],
    topo_dir: Path,
    brite_path: Path,
    *,
    save_png: bool,
) -> Dict[str, Any]:
    """Generate BRITE topology, convert to SCION, return ``scion_topo`` dict."""
    br = nested_get(cfg, "brite", default={}) or {}
    conv_cfg = nested_get(br, "scion_converter", default={}) or {}
    java = dict(nested_get(br, "java_model", default={}) or {})
    cnv = nested_get(br, "convert", default={}) or {}

    config_file = topo_dir / "brite_config.conf"
    ext = nested_get(br, "external_config_path")

    search_bases = [Path.cwd(), REPO_ROOT, REPO_ROOT / "evaluation"]

    if ext:
        src = resolve_user_path(ext, search_bases)
        print(f"\n1. Using external BRITE configuration: {src}")
        shutil.copy2(src, config_file)
        print(f"   Copied to: {config_file}")
    else:
        print("\n1. Generating BRITE configuration...")
        _eval_n = os.environ.get("EVAL_BRITE_N_NODES", "").strip()
        if _eval_n.isdigit():
            java["n_nodes"] = int(_eval_n)
            print(f"   (EVAL_BRITE_N_NODES override: n_nodes={java['n_nodes']})")
        brite_gen = BRITEConfigGenerator()
        brite_gen.generate(str(config_file), **java)
        print(f"   BRITE config saved to: {config_file}")

    print("\n2. Running BRITE...")
    brite_stem = topo_dir / "topology"
    brite_output = run_brite(Path(config_file), Path(brite_stem), brite_path=brite_path)
    print(f"   BRITE topology saved to: {brite_output}")

    root_seed = nested_get(cfg, "seed", default=None)
    extra_seed = _coalesce_int(nested_get(cnv, "extra_peering_seed"), root_seed)
    if extra_seed is None:
        extra_seed = 42

    prune_frac_raw = nested_get(cnv, "prune_cross_isd_noncore_fraction", default=0.0)
    prune_frac = float(prune_frac_raw) if prune_frac_raw is not None else 0.0
    prune_seed = _coalesce_int(
        nested_get(cnv, "prune_cross_isd_noncore_seed"),
        extra_seed,
    )

    converter = BRITE2SCIONConverter(
        n_isds=int(nested_get(conv_cfg, "n_isds", default=3)),
        core_ratio=float(nested_get(conv_cfg, "core_ratio", default=0.075)),
    )
    plot_dir = topo_dir if save_png else None
    print(f"\n3. Converting to SCION topology (plot_dir={'set' if plot_dir else 'none'})...")
    return converter.convert_brite_file(
        brite_output,
        plot_dir=plot_dir,
        extra_peering_max_links=nested_get(cnv, "extra_peering_max_links"),
        extra_peering_seed=extra_seed,
        prune_cross_isd_noncore_fraction=prune_frac,
        prune_cross_isd_noncore_seed=prune_seed,
    )
