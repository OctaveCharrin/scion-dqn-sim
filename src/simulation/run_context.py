"""Load evaluation run artifacts and construct path-selection environments."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.simulation.evaluation_env import EvaluationPathSelectionEnv, RewardWeights
from src.simulation.link_state_store import (
    LinkTrafficState,
    build_pair_path_link_idx_for_pool,
    load_link_traffic_state,
)
from src.simulation.path_store import InMemoryPathStore

TOPOLOGY_SUBDIR_NAME = "topology"

# Avoid reloading large ``link_states.pkl`` for every training script in one pipeline run.
_RUN_CONTEXT_CACHE: Dict[str, Tuple[Any, ...]] = {}


def topology_dir(run_dir: str | Path) -> Path:
    """Return ``<run_dir>/topology`` where BRITE/SCION artifacts live."""
    return Path(run_dir) / TOPOLOGY_SUBDIR_NAME


def load_topology_graph(
    run_path: Path,
) -> Tuple[nx.Graph, set[int], Dict[str, Any]]:
    """Load the SCION topology graph and core AS set from step 01 artifacts."""
    run_path = Path(run_path)
    json_path = topology_dir(run_path) / "scion_topology.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Missing {json_path}. Run 01_generate_topology.py first."
        )
    with open(json_path, "r") as f:
        topology_data = json.load(f)
    G = nx.node_link_graph(topology_data["graph"])
    core_ases = {int(x) for x in topology_data.get("core_ases", []) or []}
    return G, core_ases, topology_data


def validate_pre_training_artifacts(run_dir: str | Path) -> None:
    """Ensure topology, beaconing, and traffic outputs exist before step 04."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    topo = topology_dir(root) / "scion_topology.json"
    if not topo.is_file():
        raise FileNotFoundError(
            f"Missing {topo}. Run 01_generate_topology.py first."
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


def load_run_context(
    run_path: Path,
    *,
    use_cache: bool = True,
) -> Tuple[
    Dict[str, Any],
    InMemoryPathStore,
    LinkTrafficState,
    List[Tuple[int, int]],
    float,
]:
    """Load topology, path store, link states, pair pool, and goodput cap."""
    run_path = Path(run_path)
    cache_key = str(run_path.resolve())
    link_pkl = run_path / "link_states.pkl"
    mtime = link_pkl.stat().st_mtime if link_pkl.is_file() else 0.0
    if use_cache:
        cached = _RUN_CONTEXT_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]  # type: ignore[return-value]

    topo_json = topology_dir(run_path) / "scion_topology.json"
    if not topo_json.is_file():
        raise FileNotFoundError(
            f"Missing {topo_json}. Run 01_generate_topology.py first."
        )
    with open(topo_json, "r") as f:
        topology_data = json.load(f)
    with open(run_path / "selected_pair.json", "r") as f:
        selected_pair = json.load(f)
    path_store = InMemoryPathStore.load(run_path / "path_store.json")
    link_states = load_link_traffic_state(run_path / "link_states.pkl")

    src_as = int(selected_pair["source_as"])
    dst_as = int(selected_pair["destination_as"])
    pair_pool: List[Tuple[int, int]] = [
        (int(p[0]), int(p[1]))
        for p in selected_pair.get("pair_pool", [[src_as, dst_as]])
    ]
    if not pair_pool:
        pair_pool = [(src_as, dst_as)]

    goodput_cap = compute_goodput_cap(path_store, pair_pool)
    ctx = (topology_data, path_store, link_states, pair_pool, goodput_cap)
    if use_cache:
        _RUN_CONTEXT_CACHE[cache_key] = (mtime, ctx)
    return ctx


def compute_goodput_cap(
    path_store: InMemoryPathStore, pair_pool: Sequence[Tuple[int, int]]
) -> float:
    min_bws: List[float] = []
    for pair in pair_pool:
        for p in path_store.find_paths(int(pair[0]), int(pair[1])) or []:
            sm = p.get("static_metrics") or {}
            if "min_bandwidth" in sm:
                min_bws.append(float(sm["min_bandwidth"]))
    return max(50.0, float(np.percentile(min_bws, 95)) if min_bws else 100.0)


def compute_action_dim(
    path_store: InMemoryPathStore, selected_pair: Dict[str, Any]
) -> int:
    pair_pool = [
        (int(p[0]), int(p[1]))
        for p in selected_pair.get(
            "pair_pool", [[selected_pair["source_as"], selected_pair["destination_as"]]]
        )
    ]
    counts = [len(path_store.find_paths(int(a), int(b)) or []) for a, b in pair_pool]
    return int(
        max(
            max(counts, default=int(selected_pair.get("num_paths", 1))),
            int(selected_pair.get("max_num_paths", 1) or 1),
            int(selected_pair.get("num_paths", 1) or 1),
            1,
        )
    )


def make_env(
    topology_data: Dict[str, Any],
    path_store: InMemoryPathStore,
    link_states: LinkTrafficState,
    pair_pool: Sequence[Tuple[int, int]],
    *,
    episode_length: int = 24,
    rng_seed: Optional[int] = None,
    reward_weights: Optional[RewardWeights] = None,
    normalize_probe_penalty: bool = True,
) -> EvaluationPathSelectionEnv:
    pair_path_link_idx = None
    if not link_states.is_legacy:
        pair_path_link_idx = build_pair_path_link_idx_for_pool(
            path_store,
            pair_pool,
            link_states.link_key_to_index,
        )
    return EvaluationPathSelectionEnv(
        topology_data=topology_data,
        path_store=path_store,
        link_states=link_states,
        pair_path_link_idx=pair_path_link_idx,
        latency_probe_cost_ms=10.0,
        bandwidth_probe_cost_ms=100.0,
        per_hop_probe_cost_ms=0.5,
        per_hop_full_probe_cost_ms=20.0,
        pair_pool=list(pair_pool),
        episode_length=episode_length,
        rng_seed=rng_seed,
        reward_weights=reward_weights,
        normalize_probe_penalty=normalize_probe_penalty,
    )
