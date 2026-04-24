#!/usr/bin/env python3
"""
Generate SCION evaluation topology (BRITE + converter, or pure Python top-down).

All artifacts written under ``<run_dir>/topology/`` (JSON, pickle, plots, BRITE
inputs when applicable). A **single YAML config** selects the generator and
parameters (see ``evaluation/topology_defaults.yaml``).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import networkx as nx
import yaml

from _common import topology_dir

from src.topology.brite2scion_converter import BRITE2SCIONConverter
from src.topology.brite_cfg_gen import BRITEConfigGenerator, run_brite
from src.topology.top_down_generator import TopDownSCIONGenerator


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_REL = Path(__file__).resolve().parent / "topology_defaults.yaml"


def _get(cfg: Mapping[str, Any], *path: str, default: Any = None) -> Any:
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
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_user_path(p: Any, bases: list[Path]) -> Path:
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


def load_unified_topology_config(cli_path: Optional[Path]) -> Dict[str, Any]:
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


def _run_top_down(cfg: Dict[str, Any], topo_dir: Path, save_png: bool) -> Dict[str, Any]:
    root_seed = _get(cfg, "seed", default=None)
    td = _get(cfg, "top_down", default={}) or {}
    gp = _get(td, "geographic_peering", default={}) or {}

    gen = TopDownSCIONGenerator(seed=root_seed)
    plot_dir = topo_dir if save_png else None
    scion_topo = gen.generate(
        n_isds=int(_get(td, "n_isds", default=3)),
        n_nodes=int(_get(td, "n_nodes", default=100)),
        geographic_peering_distance_cap=_get(gp, "distance_cap"),
        geographic_peering_probability=_get(gp, "probability"),
        additional_random_peering_max_links=int(
            _get(td, "additional_random_peering_max_links", default=0)
        ),
        additional_random_peering_seed=_get(td, "additional_random_peering_seed"),
        plot_dir=plot_dir,
    )

    return scion_topo


def _run_brite(cfg: Dict[str, Any], topo_dir: Path, brite_path: Path, save_png: bool) -> Dict[str, Any]:
    br = _get(cfg, "brite", default={}) or {}
    conv_cfg = _get(br, "scion_converter", default={}) or {}
    java = dict(_get(br, "java_model", default={}) or {})
    cnv = _get(br, "convert", default={}) or {}

    config_file = topo_dir / "brite_config.conf"
    ext = _get(br, "external_config_path")

    search_bases = [Path.cwd(), REPO_ROOT, Path(__file__).resolve().parent]

    if ext:
        src = _resolve_user_path(ext, search_bases)
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

    root_seed = _get(cfg, "seed", default=None)
    extra_seed = _coalesce_int(
        _get(cnv, "extra_peering_seed"),
        root_seed,
    )
    if extra_seed is None:
        extra_seed = 42

    converter = BRITE2SCIONConverter(
        n_isds=int(_get(conv_cfg, "n_isds", default=3)),
        core_ratio=float(_get(conv_cfg, "core_ratio", default=0.075)),
    )
    plot_dir = topo_dir if save_png else None
    print(f"\n3. Converting to SCION topology (plot_dir={'set' if plot_dir else 'none'})...")
    scion_topo = converter.convert_brite_file(
        brite_output,
        plot_dir=plot_dir,
        extra_peering_max_links=_get(cnv, "extra_peering_max_links"),
        extra_peering_seed=extra_seed,
    )
    return scion_topo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SCION topology from unified YAML (BRITE or top-down)."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run directory (created if missing when omitted).",
    )
    parser.add_argument(
        "--topology-config",
        "-C",
        type=Path,
        default=None,
        help="YAML file overriding defaults (merged on top of topology_defaults.yaml).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Legacy: path to an existing BRITE .conf (overrides brite.external_config_path).",
    )
    args = parser.parse_args()

    cfg = load_unified_topology_config(args.topology_config)
    if args.config is not None:
        if "brite" not in cfg or not isinstance(cfg["brite"], dict):
            cfg["brite"] = {}
        cfg["brite"]["external_config_path"] = str(args.config)

    if args.run_dir:
        run_dir = args.run_dir
        print(f"Using run directory: {run_dir}")
    else:
        run_dir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"Using run directory: {run_dir}")

    topo_dir = topology_dir(run_dir)
    topo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Topology artifacts directory: {topo_dir}")

    if _get(cfg, "output", "dump_resolved_config", default=True):
        dump_path = topo_dir / "topology_config_resolved.yaml"
        with open(dump_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote resolved config: {dump_path}")

    save_png = bool(_get(cfg, "output", "save_step_pngs", default=True))
    generator = str(_get(cfg, "generator", default="brite")).lower().replace("-", "_")
    if generator in ("topdown", "top_down", "python", "native"):
        generator = "top_down"

    if generator == "top_down":
        print("\n=== Topology generator: top_down (pure Python) ===\n")
        scion_topo = _run_top_down(cfg, topo_dir, save_png)
    elif generator == "brite":
        print("\n=== Topology generator: brite ===\n")
        br_rel = _get(cfg, "brite", "brite_path", default="external/brite")
        br_path = Path(str(br_rel))
        if not br_path.is_absolute():
            br_path = (REPO_ROOT / br_path).resolve()
        scion_topo = _run_brite(cfg, topo_dir, br_path, save_png)
    else:
        raise SystemExit(f"Unknown generator: {generator!r} (use 'brite' or 'top_down')")

    G = scion_topo["graph"]

    topology_file = topo_dir / "scion_topology.pkl"
    with open(topology_file, "wb") as f:
        pickle.dump(scion_topo, f)
    print(f"\nSCION topology saved to: {topology_file}")

    json_data = {
        "isds": scion_topo["isds"],
        "core_ases": list(scion_topo["core_ases"]),
        "graph": nx.node_link_data(G),
    }
    json_file = topo_dir / "scion_topology.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"SCION topology JSON saved to: {json_file}")

    print("\nTopology Statistics:")
    print(f"   - Total ASes: {G.number_of_nodes()}")
    print(f"   - Total links: {G.number_of_edges()}")
    print(f"   - ISDs: {len(scion_topo['isds'])}")
    print(f"   - Core ASes: {len(scion_topo['core_ases'])}")

    link_types: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        link_type = str(data.get("type", "UNKNOWN"))
        link_types[link_type] = link_types.get(link_type, 0) + 1
    print("   - Link types:")
    for lt, count in sorted(link_types.items()):
        print(f"     - {lt}: {count}")

    if nx.is_connected(G.to_undirected()):
        print("   - Graph is connected: Yes")
    else:
        print("   - Graph is connected: No")
        components = list(nx.connected_components(G.to_undirected()))
        print(f"   - Number of components: {len(components)}")

    avg_degree = sum(dict(G.degree()).values()) / max(len(G), 1)
    print(f"   - Average degree: {avg_degree:.2f}")

    print(f"\nTopology generation complete! Artifacts in {topo_dir}/")


if __name__ == "__main__":
    main()
