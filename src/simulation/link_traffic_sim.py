"""28-day link-level traffic simulation for evaluation run directories."""

from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulation.path_store import InMemoryPathStore
from src.simulation.run_context import topology_dir

NUM_DAYS = 28
SAMPLES_PER_DAY = 24
TOTAL_HOURS = NUM_DAYS * SAMPLES_PER_DAY
BASE_RATE_MBPS = 100.0
PROP_RNG_SEED = 42

EDGE_KEY = Tuple[int, int]


def _link_key(u: int, v: int) -> EDGE_KEY:
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def _path_link_keys(path: Dict) -> List[EDGE_KEY]:
    hops = path.get("hops") or []
    keys: List[EDGE_KEY] = []
    for i in range(len(hops) - 1):
        keys.append(_link_key(hops[i]["as"], hops[i + 1]["as"]))
    return keys


def _diurnal_factor(hour: int) -> float:
    return 1.0 + 0.5 * float(np.sin((hour - 2) * np.pi / 12))


def _weekly_factor(weekday: int) -> float:
    return 0.7 if weekday >= 5 else 1.0


def _path_static_min_bw(path: Dict) -> float:
    sm = path.get("static_metrics") or {}
    val = sm.get("min_bandwidth")
    if val is not None:
        return float(val)
    return min(
        (float(h.get("bandwidth", 1e6)) for h in path.get("hops", [])), default=1e6
    )


def _path_static_latency(path: Dict) -> float:
    sm = path.get("static_metrics") or {}
    val = sm.get("total_latency")
    if val is not None:
        return float(val)
    return sum(float(h.get("latency", 10)) for h in path.get("hops", []))


def _link_metric(load_mbps: float, capacity: float, base_lat: float):
    cap = max(1.0, float(capacity))
    util = min(1.5, max(0.0, float(load_mbps) / cap))
    util_clipped = min(0.99, util)
    queue_mult = 1.0 + 1.0 * (util_clipped / max(0.01, 1.0 - util_clipped)) * 0.05
    queue_mult = min(2.0, queue_mult)
    latency = float(base_lat) * queue_mult
    available = max(0.0, cap - float(load_mbps))
    if util >= 0.95:
        loss = min(0.2, 0.001 + 0.05 * (util - 0.95) / 0.05)
    elif util >= 0.8:
        loss = 0.001 + 0.005 * (util - 0.8) / 0.15
    else:
        loss = 0.0
    return latency, available, util, loss


def simulate_link_traffic(run_path: Path) -> Dict:
    """Run the full traffic simulation and persist artifacts under ``run_path``.

    Returns the simulation metadata dict written to ``simulation_metadata.json``.
    """
    run_path = Path(run_path)

    topo_json = topology_dir(run_path) / "scion_topology.json"
    if not topo_json.is_file():
        legacy = run_path / "scion_topology.json"
        topo_json = legacy if legacy.is_file() else topo_json
    with open(topo_json, "r") as f:
        topology_data = json.load(f)

    with open(run_path / "selected_pair.json", "r") as f:
        selected_pair = json.load(f)

    path_store = InMemoryPathStore.load(run_path / "path_store.json")

    pkl_paths = [
        topology_dir(run_path) / "scion_topology.pkl",
        run_path / "scion_topology.pkl",
    ]
    G = None
    for cand in pkl_paths:
        if cand.is_file():
            with open(cand, "rb") as f:
                scion_topo = pickle.load(f)
                if isinstance(scion_topo.get("graph"), nx.Graph):
                    G = scion_topo["graph"]
                    break
    if G is None:
        G = nx.node_link_graph(topology_data["graph"])

    src_as = int(selected_pair["source_as"])
    dst_as = int(selected_pair["destination_as"])

    pair_pool: List[Tuple[int, int]] = [
        (int(p[0]), int(p[1])) for p in selected_pair.get("pair_pool", [[src_as, dst_as]])
    ]
    if not pair_pool:
        pair_pool = [(src_as, dst_as)]

    print(f"\nSimulating traffic for {len(pair_pool)} foreground AS pair(s)")
    print(f"  Selected legacy pair: {src_as} -> {dst_as}")

    n_nodes = G.number_of_nodes()
    background_pairs_per_hour = max(20, min(200, 4 * n_nodes))

    link_capacity: Dict[EDGE_KEY, float] = {}
    link_base_latency: Dict[EDGE_KEY, float] = {}
    for u, v, data in G.edges(data=True):
        k = _link_key(u, v)
        link_capacity[k] = float(data.get("bandwidth", data.get("capacity", 1000.0)))
        link_base_latency[k] = float(data.get("latency", data.get("delay", 10.0)))

    print(
        f"  Topology: {n_nodes} ASes, {len(link_capacity)} links, "
        f"avg link capacity {np.mean(list(link_capacity.values())):.0f} Mbps"
    )

    paths_by_pair: Dict[Tuple[int, int], List[Dict]] = {}
    for pair in pair_pool:
        plist = path_store.find_paths(int(pair[0]), int(pair[1]))
        if plist:
            paths_by_pair[pair] = plist

    all_pairs: List[Tuple[int, int]] = []
    seen = set()
    for pair in pair_pool:
        if pair in paths_by_pair and pair not in seen:
            all_pairs.append(pair)
            seen.add(pair)
    try:
        for k in path_store._paths.keys():  # type: ignore[attr-defined]
            if k in seen:
                continue
            if path_store.find_paths(*k):
                all_pairs.append(k)
                seen.add(k)
    except Exception:
        pass

    print(f"  Foreground pairs with paths: {len(paths_by_pair)}")
    print(
        f"  Routable pairs available for background draw: "
        f"{len(all_pairs)} (of {n_nodes * (n_nodes - 1)})"
    )

    print("\nGenerating foreground traffic series (28 days x 24 hours)...")
    np.random.seed(42)

    flows: List[Dict] = []
    selected_flow_series: List[Dict] = []

    start_time = datetime.now()
    for pair in pair_pool:
        pair_seed = (hash(pair) ^ 0xABCDEF) & 0x7FFFFFFF
        pair_rng = np.random.default_rng(pair_seed)
        for day in range(NUM_DAYS):
            for hour in range(24):
                timestamp = start_time + timedelta(days=day, hours=hour)
                wd = timestamp.weekday()
                base = BASE_RATE_MBPS * _diurnal_factor(hour) * _weekly_factor(wd)
                jitter = float(pair_rng.uniform(0.8, 1.2))
                mbps = float(base * jitter)
                flow = {
                    "timestamp": timestamp,
                    "source_as": int(pair[0]),
                    "destination_as": int(pair[1]),
                    "bandwidth_mbps": mbps,
                    "duration_s": 3600,
                    "day": day,
                    "hour": hour,
                    "day_of_week": wd,
                }
                flows.append(flow)
                if pair == (src_as, dst_as):
                    selected_flow_series.append(flow)

    print(f"  Generated {len(flows)} foreground flow samples")

    traffic_file = run_path / "traffic_flows.pkl"
    with open(traffic_file, "wb") as f:
        pickle.dump(selected_flow_series, f)
    df = pd.DataFrame(selected_flow_series)
    df.to_csv(run_path / "traffic_flows.csv", index=False)
    print(f"  Saved selected-pair flows: {traffic_file}")

    all_flows_file = run_path / "traffic_flows_all_pairs.pkl"
    with open(all_flows_file, "wb") as f:
        pickle.dump(flows, f)
    print(f"  Saved all-pairs flows:    {all_flows_file}")

    print("\nAggregating link loads (with background traffic)...")
    bg_rng = random.Random(PROP_RNG_SEED + 7)

    def draw_background_pairs(n: int) -> List[Tuple[int, int]]:
        if not all_pairs:
            return []
        return [bg_rng.choice(all_pairs) for _ in range(n)]

    def route_flow_to_path(pair: Tuple[int, int]) -> List[EDGE_KEY] | None:
        plist = paths_by_pair.get(pair)
        if plist is None:
            plist = path_store.find_paths(int(pair[0]), int(pair[1]))
        if not plist:
            return None
        best = min(plist, key=_path_static_latency)
        return _path_link_keys(best)

    link_loads_by_hour: Dict[int, Dict[EDGE_KEY, float]] = {}

    for h in tqdm(range(TOTAL_HOURS), desc="hours", ncols=80):
        timestamp = start_time + timedelta(hours=h)
        wd = timestamp.weekday()
        diurnal = _diurnal_factor(h % 24)
        weekly = _weekly_factor(wd)
        hour_load: Dict[EDGE_KEY, float] = defaultdict(float)

        for pair, plist in paths_by_pair.items():
            pair_seed = (hash(pair) ^ 0xABCDEF) & 0x7FFFFFFF
            pair_rng = np.random.default_rng(pair_seed + h)
            mbps = BASE_RATE_MBPS * diurnal * weekly * float(pair_rng.uniform(0.8, 1.2))
            ranked = sorted(plist, key=_path_static_latency)[:3]
            weights = np.array(
                [1.0 / max(1.0, _path_static_latency(p)) for p in ranked], dtype=float
            )
            if weights.sum() == 0:
                continue
            weights /= weights.sum()
            for w, p in zip(weights, ranked):
                for k in _path_link_keys(p):
                    hour_load[k] += float(w * mbps)

        bg_pairs = draw_background_pairs(background_pairs_per_hour)
        bg_factor = diurnal * weekly
        for bp in bg_pairs:
            keys = route_flow_to_path(bp)
            if not keys:
                continue
            bg_mbps = float(bg_rng.uniform(10.0, 80.0)) * bg_factor
            for k in keys:
                hour_load[k] += bg_mbps

        link_loads_by_hour[h] = dict(hour_load)

    print("\nDeriving per-path metrics from aggregate link load...")

    selected_pair_paths = paths_by_pair.get((src_as, dst_as)) or path_store.find_paths(
        src_as, dst_as
    )

    link_states: Dict[int, Dict] = {}

    for h in range(TOTAL_HOURS):
        hour_loads = link_loads_by_hour.get(h, {})
        link_metric_cache: Dict[EDGE_KEY, Tuple[float, float, float, float]] = {}

        def resolve(k: EDGE_KEY):
            if k not in link_metric_cache:
                cap = link_capacity.get(k, 1000.0)
                base = link_base_latency.get(k, 10.0)
                link_metric_cache[k] = _link_metric(hour_loads.get(k, 0.0), cap, base)
            return link_metric_cache[k]

        hour_state: Dict[str, Dict] = {}
        by_pair_state: Dict[str, Dict] = {}
        for pair, plist in paths_by_pair.items():
            per_pair: Dict[str, Dict] = {}
            for path_idx, path in enumerate(plist):
                keys = _path_link_keys(path)
                if not keys:
                    continue
                metrics = [resolve(k) for k in keys]
                latencies = [m[0] for m in metrics]
                avails = [m[1] for m in metrics]
                utils = [m[2] for m in metrics]
                losses = [m[3] for m in metrics]
                total_lat = float(sum(latencies))
                bottleneck_avail = float(min(avails))
                max_util = float(max(utils))
                survive = float(np.prod([1.0 - x for x in losses]))
                agg_loss = float(1.0 - survive)
                per_pair[f"path_{path_idx}"] = {
                    "latency_ms": total_lat,
                    "available_bandwidth_mbps": bottleneck_avail,
                    "utilization": max_util,
                    "loss_rate": agg_loss,
                    "hop_count": len(keys),
                }
            by_pair_state[f"pair_{int(pair[0])}_{int(pair[1])}"] = per_pair

        hour_state["by_pair"] = by_pair_state
        sel_key = f"pair_{src_as}_{dst_as}"
        sel_block = by_pair_state.get(sel_key, {})
        for k, v in sel_block.items():
            hour_state[k] = v

        link_states[h] = hour_state

    link_states_file = run_path / "link_states.pkl"
    with open(link_states_file, "wb") as f:
        pickle.dump(link_states, f)
    print(f"Link states saved to: {link_states_file}")
    link_states_json = run_path / "link_states.json"
    with open(link_states_json, "w") as f:
        json.dump(link_states, f)
    print(f"Link states json saved to: {link_states_json}")

    print("\nTraffic simulation summary:")
    print(f"  - Total days:              {NUM_DAYS}")
    print(f"  - Foreground pairs:        {len(paths_by_pair)}")
    print(f"  - Background pairs/hour:   {background_pairs_per_hour}")
    print(f"  - Total foreground flows:  {len(flows)}")
    print(
        f"  - Average bandwidth (sel): {np.mean([f['bandwidth_mbps'] for f in selected_flow_series]):.2f} Mbps"
    )
    print(f"  - Training period: Days 1-14 ({14 * SAMPLES_PER_DAY} samples)")
    print(f"  - Evaluation period: Days 15-28 ({14 * SAMPLES_PER_DAY} samples)")

    metadata = {
        "source_as": src_as,
        "destination_as": dst_as,
        "num_pairs": len(paths_by_pair),
        "num_paths": len(selected_pair_paths or []),
        "num_days": NUM_DAYS,
        "samples_per_day": SAMPLES_PER_DAY,
        "total_samples": TOTAL_HOURS,
        "training_samples": 14 * SAMPLES_PER_DAY,
        "evaluation_samples": 14 * SAMPLES_PER_DAY,
        "background_pairs_per_hour": background_pairs_per_hour,
        "traffic_stats": {
            "mean_bandwidth_mbps": float(df["bandwidth_mbps"].mean()),
            "std_bandwidth_mbps": float(df["bandwidth_mbps"].std()),
            "min_bandwidth_mbps": float(df["bandwidth_mbps"].min()),
            "max_bandwidth_mbps": float(df["bandwidth_mbps"].max()),
        },
    }

    with open(run_path / "simulation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSimulation metadata saved to: {run_path / 'simulation_metadata.json'}")
    print("\nTraffic simulation complete!")
    return metadata
