"""28-day link-level traffic simulation for evaluation run directories."""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulation.path_store import InMemoryPathStore
from src.simulation.run_context import load_topology_graph
from src.simulation.traffic_config import TrafficSimConfig
from src.simulation.traffic_metrics import (
    compute_link_metrics_vectorized,
    path_metrics_from_link_indices,
    summarize_link_loads,
)

EDGE_KEY = Tuple[int, int]


def _link_key(u: int, v: int) -> EDGE_KEY:
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def _path_link_keys(path: Dict) -> List[EDGE_KEY]:
    hops = path.get("hops") or []
    return [_link_key(hops[i]["as"], hops[i + 1]["as"]) for i in range(len(hops) - 1)]


def _diurnal_factor(hour: int) -> float:
    return 1.0 + 0.5 * float(np.sin((hour - 2) * np.pi / 12))


def _weekly_factor(weekday: int) -> float:
    return 0.7 if weekday >= 5 else 1.0


def _path_static_latency(path: Dict) -> float:
    sm = path.get("static_metrics") or {}
    val = sm.get("total_latency")
    if val is not None:
        return float(val)
    return sum(float(h.get("latency", 10)) for h in path.get("hops", []))


@dataclass
class _PairRouting:
    """Precomputed top-path ECMP split for one AS pair."""

    pair: Tuple[int, int]
    path_link_idx: List[List[int]]
    weights: np.ndarray


def _build_pair_routings(
    pair: Tuple[int, int],
    plist: List[Dict],
    link_key_to_index: Dict[EDGE_KEY, int],
    top_k: int,
) -> _PairRouting | None:
    ranked = sorted(plist, key=_path_static_latency)[:top_k]
    if not ranked:
        return None
    weights = np.array(
        [1.0 / max(1.0, _path_static_latency(p)) for p in ranked], dtype=np.float64
    )
    if weights.sum() <= 0:
        return None
    weights /= weights.sum()
    path_link_idx: List[List[int]] = []
    for p in ranked:
        idx = [
            link_key_to_index[k] for k in _path_link_keys(p) if k in link_key_to_index
        ]
        path_link_idx.append(idx)
    return _PairRouting(pair=pair, path_link_idx=path_link_idx, weights=weights)


def _pair_gravity_weights(
    G: nx.Graph, pairs: Sequence[Tuple[int, int]], rng: np.random.Generator
) -> np.ndarray:
    """Sampling weights ~ sqrt(deg(src)*deg(dst)) for active-pair selection."""
    deg = dict(G.degree())
    w = np.array(
        [
            float(np.sqrt(max(1, deg.get(a, 1)) * max(1, deg.get(b, 1))))
            for a, b in pairs
        ],
        dtype=np.float64,
    )
    if w.sum() <= 0:
        w = np.ones(len(pairs), dtype=np.float64)
    return w / w.sum()


def _sample_active_pairs(
    pair_list: List[Tuple[int, int]],
    G: nx.Graph,
    n_active: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if n_active >= len(pair_list):
        return list(pair_list)
    weights = _pair_gravity_weights(G, pair_list, rng)
    idx = rng.choice(len(pair_list), size=n_active, replace=False, p=weights)
    return [pair_list[int(i)] for i in idx]


def simulate_link_traffic(
    run_path: Path,
    config: TrafficSimConfig | None = None,
) -> Dict:
    """Run traffic simulation and persist artifacts under ``run_path``."""
    run_path = Path(run_path)
    cfg = config or TrafficSimConfig.from_env()

    G, _core_ases, _topology_data = load_topology_graph(run_path)

    with open(run_path / "selected_pair.json") as f:
        selected_pair = json.load(f)

    path_store = InMemoryPathStore.load(run_path / "path_store.json")

    src_as = int(selected_pair["source_as"])
    dst_as = int(selected_pair["destination_as"])

    pair_pool: List[Tuple[int, int]] = [
        (int(p[0]), int(p[1]))
        for p in selected_pair.get("pair_pool", [[src_as, dst_as]])
    ]
    if not pair_pool:
        pair_pool = [(src_as, dst_as)]

    n_nodes = G.number_of_nodes()
    pool_size = len(pair_pool)
    scaled_base = cfg.scaled_base_rate_mbps(pool_size)

    print("\nSimulating traffic (calibrated demand model)")
    print(f"  Pair pool size: {pool_size}")
    print(f"  Reference pair: {src_as} -> {dst_as}")
    print(
        f"  Scaled base rate: {scaled_base:.2f} Mbps (ref pool={cfg.reference_pair_pool_size})"
    )
    print(
        f"  Active foreground pairs/hour: {cfg.active_pairs_min}–{cfg.active_pairs_max}"
    )

    link_keys: List[EDGE_KEY] = []
    link_capacity_list: List[float] = []
    link_latency_list: List[float] = []
    link_key_to_index: Dict[EDGE_KEY, int] = {}
    for u, v, data in G.edges(data=True):
        k = _link_key(u, v)
        if k not in link_key_to_index:
            link_key_to_index[k] = len(link_keys)
            link_keys.append(k)
            link_capacity_list.append(
                float(data.get("bandwidth", data.get("capacity", 1000.0)))
            )
            link_latency_list.append(
                float(data.get("latency", data.get("delay", 10.0)))
            )

    n_links = len(link_keys)
    capacities = np.array(link_capacity_list, dtype=np.float64)
    base_lats = np.array(link_latency_list, dtype=np.float64)

    print(
        f"  Topology: {n_nodes} ASes, {n_links} links, "
        f"avg capacity {float(capacities.mean()):.0f} Mbps"
    )

    paths_by_pair: Dict[Tuple[int, int], List[Dict]] = {}
    for pair in pair_pool:
        plist = path_store.find_paths(int(pair[0]), int(pair[1]))
        if plist:
            paths_by_pair[pair] = plist

    routable_pairs = list(paths_by_pair.keys())
    routings: Dict[Tuple[int, int], _PairRouting] = {}
    for pair, plist in paths_by_pair.items():
        pr = _build_pair_routings(
            pair, plist, link_key_to_index, cfg.top_paths_per_pair
        )
        if pr is not None:
            routings[pair] = pr

    bg_pair_list = routable_pairs

    print(f"  Routable pairs: {len(routable_pairs)}")
    print("\nPrecomputing foreground flow table (sparse active pairs)...")

    np.random.seed(cfg.prop_rng_seed)
    hour_rng = np.random.default_rng(cfg.prop_rng_seed)
    pair_flow_rng: Dict[Tuple[int, int], np.random.Generator] = {}

    n_active_typical = min(
        cfg.active_pairs_max, max(cfg.active_pairs_min, len(routable_pairs))
    )
    bg_per_hour_typical = cfg.background_pairs_per_hour(n_nodes, n_active_typical)

    flows: List[Dict] = []
    selected_flow_series: List[Dict] = []
    active_schedule: Dict[int, List[Tuple[Tuple[int, int], float]]] = {}

    start_time = datetime.now()
    total_hours = cfg.total_hours

    for h in range(total_hours):
        hour_of_day = h % 24
        day = h // 24
        timestamp = start_time + timedelta(hours=h)
        wd = timestamp.weekday()
        diurnal = _diurnal_factor(hour_of_day)
        weekly = _weekly_factor(wd)

        n_active = int(
            hour_rng.integers(cfg.active_pairs_min, cfg.active_pairs_max + 1)
        )
        n_active = min(n_active, len(routable_pairs))
        active_pairs = _sample_active_pairs(routable_pairs, G, n_active, hour_rng)
        n_elephants = max(1, int(len(active_pairs) * cfg.elephant_fraction))
        elephant_set = set(
            hour_rng.choice(
                len(active_pairs),
                size=min(n_elephants, len(active_pairs)),
                replace=False,
            )
        )

        hour_entries: List[Tuple[Tuple[int, int], float]] = []
        for i, pair in enumerate(active_pairs):
            if pair not in pair_flow_rng:
                seed = (hash(pair) ^ 0xABCDEF) & 0x7FFFFFFF
                pair_flow_rng[pair] = np.random.default_rng(seed)

            rate_mult = cfg.elephant_rate_multiplier if i in elephant_set else 1.0
            jitter = float(pair_flow_rng[pair].uniform(0.8, 1.2))
            mbps = float(scaled_base * diurnal * weekly * rate_mult * jitter)

            flow = {
                "timestamp": timestamp,
                "source_as": int(pair[0]),
                "destination_as": int(pair[1]),
                "bandwidth_mbps": mbps,
                "duration_s": 3600,
                "day": day,
                "hour": hour_of_day,
                "day_of_week": wd,
                "active_hour": h,
            }
            flows.append(flow)
            hour_entries.append((pair, mbps))
            if pair == (src_as, dst_as):
                selected_flow_series.append(flow)

        active_schedule[h] = hour_entries

    print(f"  Recorded {len(flows)} foreground flow entries (sparse schedule)")

    with open(run_path / "traffic_flows.pkl", "wb") as f:
        pickle.dump(selected_flow_series, f)
    pd.DataFrame(selected_flow_series).to_csv(
        run_path / "traffic_flows.csv", index=False
    )
    with open(run_path / "traffic_flows_all_pairs.pkl", "wb") as f:
        pickle.dump(flows, f)

    print("\nAggregating link loads (vectorized)...")
    bg_rng = random.Random(cfg.prop_rng_seed + 7)
    link_loads_by_hour: Dict[int, np.ndarray] = {}
    peak_hour = 14 + 14 * 24

    for h in tqdm(range(total_hours), desc="hours", ncols=80):
        hour_of_day = h % 24
        diurnal = _diurnal_factor(hour_of_day)
        weekly = _weekly_factor((start_time + timedelta(hours=h)).weekday())
        loads = np.zeros(n_links, dtype=np.float64)

        hour_entries = active_schedule[h]
        bg_n = cfg.background_pairs_per_hour(n_nodes, len(hour_entries))

        for pair, mbps in hour_entries:
            pr = routings.get(pair)
            if pr is None:
                continue
            for w, link_idxs in zip(pr.weights, pr.path_link_idx):
                if link_idxs:
                    loads[link_idxs] += float(w * mbps)

        if bg_pair_list and bg_n > 0:
            bg_weights = _pair_gravity_weights(
                G, bg_pair_list, np.random.default_rng(cfg.prop_rng_seed + h)
            )
            bg_indices = bg_rng.choices(
                range(len(bg_pair_list)), k=bg_n, weights=bg_weights.tolist()
            )
            bg_factor = diurnal * weekly
            for bi in bg_indices:
                pair = bg_pair_list[bi]
                pr = routings.get(pair)
                if pr is None or not pr.path_link_idx:
                    continue
                bg_mbps = (
                    float(
                        bg_rng.uniform(cfg.background_mbps_min, cfg.background_mbps_max)
                    )
                    * bg_factor
                )
                loads[pr.path_link_idx[0]] += bg_mbps

        link_loads_by_hour[h] = loads

    print("\nDeriving per-path metrics...")
    link_states: Dict[int, Dict] = {}
    pair_list = list(paths_by_pair.keys())
    pair_path_link_idx: Dict[Tuple[int, int], List[List[int]]] = {}
    for pair, plist in paths_by_pair.items():
        pair_path_link_idx[pair] = [
            [link_key_to_index[k] for k in _path_link_keys(p) if k in link_key_to_index]
            for p in plist
        ]

    for h in tqdm(range(total_hours), desc="link_states", ncols=80):
        lm = compute_link_metrics_vectorized(
            link_loads_by_hour[h],
            capacities,
            base_lats,
            util_cap=cfg.util_cap_in_path_metrics,
        )
        by_pair_state: Dict[str, Dict] = {}
        for pair in pair_list:
            per_pair: Dict[str, Dict] = {}
            for path_idx, keys in enumerate(pair_path_link_idx[pair]):
                per_pair[f"path_{path_idx}"] = path_metrics_from_link_indices(keys, lm)
            by_pair_state[f"pair_{int(pair[0])}_{int(pair[1])}"] = per_pair

        hour_state: Dict[str, Dict] = {"by_pair": by_pair_state}
        sel_key = f"pair_{src_as}_{dst_as}"
        sel_block = by_pair_state.get(sel_key, {})
        for k, v in sel_block.items():
            hour_state[k] = v
        link_states[h] = hour_state

    with open(run_path / "link_states.pkl", "wb") as f:
        pickle.dump(link_states, f)
    print(f"Wrote {run_path / 'link_states.pkl'}")

    if cfg.write_link_states_json:
        with open(run_path / "link_states.json", "w") as f:
            json.dump(link_states, f)
        print(f"Wrote {run_path / 'link_states.json'}")

    util_summary = summarize_link_loads(
        link_loads_by_hour, capacities, peak_hour=peak_hour
    )

    # Quick path-quality sample on eval window
    eval_hours = list(range(14 * 24, total_hours))
    sample_pairs = pair_list[: min(64, len(pair_list))]
    zero_ph = total_ph = 0
    for h in eval_hours[:: max(1, len(eval_hours) // 48)]:
        block = link_states[h].get("by_pair", {})
        for pair in sample_pairs:
            key = f"pair_{pair[0]}_{pair[1]}"
            per = block.get(key, {})
            if not per:
                continue
            bws = [float(v.get("available_bandwidth_mbps", 0)) for v in per.values()]
            if not bws:
                continue
            total_ph += 1
            if max(bws) <= 0.001:
                zero_ph += 1

    print("\nTraffic simulation summary:")
    print(f"  Foreground flows recorded: {len(flows)}")
    print(f"  Pairs with path metrics:   {len(pair_list)}")
    print(f"  Background pairs/hour:     ~{bg_per_hour_typical}")
    if util_summary:
        print(
            f"  Peak-hour link util (raw p90): "
            f"{util_summary.get('util_peak_hour_raw_p90', 0):.3f}"
        )
        print(
            f"  Peak-hour link util (raw max): "
            f"{util_summary.get('util_peak_hour_raw_max', 0):.3f}"
        )
        print(
            f"  Fraction links util>1:     "
            f"{util_summary.get('fraction_links_util_gt_1', 0):.1%}"
        )
    if total_ph:
        print(
            f"  Sampled zero max-path-BW (pair,h): "
            f"{zero_ph / total_ph:.1%} ({zero_ph}/{total_ph})"
        )

    selected_pair_paths = paths_by_pair.get((src_as, dst_as)) or []

    metadata = {
        "source_as": src_as,
        "destination_as": dst_as,
        "num_pairs": len(pair_list),
        "num_paths": len(selected_pair_paths or []),
        "num_days": cfg.num_days,
        "samples_per_day": cfg.samples_per_day,
        "total_samples": total_hours,
        "training_samples": 14 * cfg.samples_per_day,
        "evaluation_samples": 14 * cfg.samples_per_day,
        "traffic_calibration": {
            **cfg.to_dict(),
            "scaled_base_rate_mbps": scaled_base,
            "pair_pool_size": pool_size,
            "background_pairs_per_hour_typical": bg_per_hour_typical,
            "foreground_flows_total": len(flows),
        },
        "link_utilization": util_summary,
        "path_quality_sample": {
            "fraction_max_path_bw_zero": float(zero_ph / total_ph) if total_ph else 0.0,
            "pair_hours_sampled": total_ph,
        },
        "traffic_stats": {
            "mean_bandwidth_mbps": float(
                np.mean([f["bandwidth_mbps"] for f in selected_flow_series])
            )
            if selected_flow_series
            else 0.0,
            "std_bandwidth_mbps": float(
                np.std([f["bandwidth_mbps"] for f in selected_flow_series])
            )
            if selected_flow_series
            else 0.0,
        },
    }

    with open(run_path / "simulation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    inspect_path = run_path / "traffic_inspection.json"
    from src.simulation.traffic_inspect import analyze_traffic_run

    report = analyze_traffic_run(run_path)
    with open(inspect_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {inspect_path}")

    print("\nTraffic simulation complete!")
    return metadata
