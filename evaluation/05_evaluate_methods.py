#!/usr/bin/env python3
"""Evaluate path-selection methods on the last 14 days.

Mirrors the training pipeline:

* iterates over the **multi-pair** ``pair_pool`` from the beaconing step,
* uses the same state featurization (no 0.5/0.7 placeholders),
* attributes probe overhead via ``env.last_probe_cost_ms`` / ``total_probe_cost_ms``
  rather than recomputing it from a different formula in the harness.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from _common import resolve_run_dir, topology_dir

from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.simulation.evaluation_env import EvaluationPathSelectionEnv
from src.baselines.shortest_path import ShortestPathSelector
from src.baselines.widest_path import WidestPathSelector
from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.ecmp import ECMPSelector
from src.baselines.random_selection import RandomSelector
from src.baselines.scion_default import SCIONDefaultSelector

run_dir = resolve_run_dir()
run_path = Path(run_dir)


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

_topo_json = topology_dir(run_path) / "scion_topology.json"
if not _topo_json.is_file():
    _leg = run_path / "scion_topology.json"
    _topo_json = _leg if _leg.is_file() else _topo_json
with open(_topo_json, "r") as f:
    topology_data = json.load(f)

with open(run_path / "selected_pair.json", "r") as f:
    selected_pair = json.load(f)

with open(run_path / "path_store.pkl", "rb") as f:
    path_store = pickle.load(f)

with open(run_path / "traffic_flows.pkl", "rb") as f:
    traffic_flows = pickle.load(f)

with open(run_path / "link_states.pkl", "rb") as f:
    link_states = pickle.load(f)

# Trained DQN checkpoint (from 04_train_dqn.py).
_model_path = run_path / "dqn_model.pth"
try:
    model_checkpoint = torch.load(_model_path, map_location="cpu", weights_only=False)
except TypeError:
    model_checkpoint = torch.load(_model_path, map_location="cpu")

src_as = int(selected_pair["source_as"])
dst_as = int(selected_pair["destination_as"])

pair_pool: List[Tuple[int, int]] = [
    (int(p[0]), int(p[1]))
    for p in selected_pair.get("pair_pool", [[src_as, dst_as]])
]
if not pair_pool:
    pair_pool = [(src_as, dst_as)]

EVAL_PAIRS = pair_pool[: min(len(pair_pool), 32)]  # cap to keep eval bounded
print(f"\nEvaluating across {len(EVAL_PAIRS)} pair(s) (of {len(pair_pool)} in pool)")


# Hours 14*24..28*24 form the evaluation window.
EVAL_HOURS = list(range(14 * 24, 28 * 24))
print(f"Evaluation horizon: {len(EVAL_HOURS)} hours x {len(EVAL_PAIRS)} pairs")


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

env = EvaluationPathSelectionEnv(
    topology_data=topology_data,
    path_store=path_store,
    link_states=link_states,
    latency_probe_cost_ms=10.0,
    bandwidth_probe_cost_ms=100.0,
    per_hop_probe_cost_ms=0.5,
    per_hop_full_probe_cost_ms=20.0,
    pair_pool=EVAL_PAIRS,
    episode_length=1,
    rng_seed=7,
)


# -----------------------------------------------------------------------------
# DQN agent (reload checkpoint)
# -----------------------------------------------------------------------------

_config: EnhancedDQNConfig = model_checkpoint["config"]
_state_dim = int(model_checkpoint.get("state_dim", 5))
_action_dim = int(model_checkpoint.get("action_dim", selected_pair.get("num_paths", 1)))
GOODPUT_CAP_MBPS = float(model_checkpoint.get("goodput_cap_mbps", 100.0))

dqn_agent = EnhancedDQNAgent(_state_dim, _action_dim, _config)
_q = model_checkpoint.get("q_network") or model_checkpoint.get("model_state_dict")
dqn_agent.q_network.load_state_dict(_q)
if "target_network" in model_checkpoint:
    dqn_agent.target_network.load_state_dict(model_checkpoint["target_network"])
dqn_agent.epsilon = 0.0  # No exploration during evaluation


# -----------------------------------------------------------------------------
# Reward & state featurization (must match 04_train_dqn.py)
# -----------------------------------------------------------------------------

w1, w2, w3, w4 = 0.7, 0.3, 0.5, 0.5
_rw = model_checkpoint.get("reward_weights") or {}
w1 = float(_rw.get("w1", w1))
w2 = float(_rw.get("w2", w2))
w3 = float(_rw.get("w3", w3))
w4 = float(_rw.get("w4", w4))


def _aggregate_state(env: EvaluationPathSelectionEnv, hour_idx: int) -> np.ndarray:
    day = (hour_idx // 24) % 7
    hour = hour_idx % 24
    f0 = day / 6.0
    f1 = hour / 23.0
    states = list(env.current_link_states.values())
    if not states:
        return np.array([f0, f1, 0.0, 0.0, 0.0], dtype=np.float32)
    utils = [float(s.get("utilization", 0.0)) for s in states]
    losses = [float(s.get("loss_rate", 0.0)) for s in states]
    lats = [
        min(100.0, float(s.get("latency_ms", 50.0))) / 100.0 for s in states
    ]
    trusts = [
        max(0.0, min(1.0, 1.0 - (w3 * loss + w4 * lat)))
        for loss, lat in zip(losses, lats)
    ]
    f2 = float(np.mean(utils)) if utils else 0.0
    f3 = float(np.mean(trusts)) if trusts else 0.0
    f4 = float(np.mean([1.0 if u > 0.7 else 0.0 for u in utils])) if utils else 0.0
    return np.array([f0, f1, f2, f3, f4], dtype=np.float32)


def _compute_reward(path_metrics: Dict) -> float:
    bw = float(path_metrics.get("bandwidth_mbps") or 0.0)
    goodput = max(0.0, min(bw / GOODPUT_CAP_MBPS, 1.0))
    loss = float(path_metrics.get("loss_rate", 0.0))
    delay = min(100.0, float(path_metrics.get("latency_ms", 50.0))) / 100.0
    trust = max(0.0, min(1.0, 1.0 - (w3 * loss + w4 * delay)))
    return float(2.0 * (w1 * goodput + w2 * trust) - 1.0)


# -----------------------------------------------------------------------------
# Methods under evaluation
# -----------------------------------------------------------------------------

baseline_methods = {
    "shortest_path": ShortestPathSelector(),
    "widest_path": WidestPathSelector(),
    "lowest_latency": LowestLatencySelector(),
    "ecmp": ECMPSelector(),
    "random": RandomSelector(),
    "scion_default": SCIONDefaultSelector(),
}


results: Dict[str, Dict] = defaultdict(
    lambda: {
        "rewards": [],
        "latencies": [],
        "bandwidths": [],
        "losses": [],
        "latency_probes": 0,
        "bandwidth_probes": 0,
        "total_probe_time_ms": 0.0,
        "selection_times_ms": [],
    }
)

print("\nEvaluating methods...")

for method_name, method in [("dqn", dqn_agent)] + list(baseline_methods.items()):
    print(f"\n--- {method_name} ---")
    method_results = results[method_name]

    pbar = tqdm(
        total=len(EVAL_HOURS) * len(EVAL_PAIRS),
        desc=method_name,
        ncols=80,
    )
    for hour_idx in EVAL_HOURS:
        for pair in EVAL_PAIRS:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            paths = env.available_paths
            if not paths:
                pbar.update(1)
                continue

            start_time = time.time()
            path_metrics: Dict = {}

            if method_name == "dqn":
                state = _aggregate_state(env, hour_idx)
                mask = env.action_mask(_action_dim)
                action = int(dqn_agent.act(state, action_mask=mask))
                if action >= len(paths):
                    valid = np.where(mask)[0]
                    action = int(valid[0]) if len(valid) > 0 else 0
                # Selective probing: probe only the chosen path.
                path_metrics = env.probe_path_full(action)
                method_results["bandwidth_probes"] += 1
                method_results["latency_probes"] += 1
                method_results["total_probe_time_ms"] += env.last_probe_cost_ms
            else:
                # Baselines probe every path.
                path_metrics_list: List[Dict] = []
                for path_idx in range(len(paths)):
                    if method_name in ("widest_path", "ecmp"):
                        m = env.probe_path_full(path_idx)
                        method_results["latency_probes"] += 1
                        method_results["bandwidth_probes"] += 1
                    else:
                        m = env.probe_path_latency(path_idx)
                        method_results["latency_probes"] += 1
                    method_results["total_probe_time_ms"] += env.last_probe_cost_ms
                    path_metrics_list.append(m)

                flow_stub = {"src": int(pair[0]), "dst": int(pair[1])}
                state_stub = np.zeros(1, dtype=np.float32)
                if method_name == "random":
                    action = int(np.random.choice(len(paths)))
                else:
                    action = int(method.select_path(paths, path_metrics_list, flow_stub, state_stub))
                if 0 <= action < len(paths):
                    path_metrics = path_metrics_list[action]

            selection_time_ms = (time.time() - start_time) * 1000.0
            method_results["selection_times_ms"].append(selection_time_ms)

            if not path_metrics:
                pbar.update(1)
                continue

            latency = float(path_metrics.get("latency_ms", 50.0))
            bw_val = path_metrics.get("bandwidth_mbps")
            bandwidth = float(bw_val) if bw_val is not None else 0.0
            loss_rate = float(path_metrics.get("loss_rate", 0.0))

            method_results["rewards"].append(_compute_reward(path_metrics))
            method_results["latencies"].append(latency)
            method_results["bandwidths"].append(bandwidth)
            method_results["losses"].append(loss_rate)
            pbar.update(1)
    pbar.close()


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

summary: Dict[str, Dict] = {}
for method_name, method_results in results.items():
    if not method_results["rewards"]:
        continue
    rewards = np.array(method_results["rewards"])
    latencies = np.array(method_results["latencies"])
    bandwidths = np.array(method_results["bandwidths"])
    selection_times = np.array(method_results["selection_times_ms"])
    probe_time = float(method_results["total_probe_time_ms"])
    n_selections = max(1, len(method_results["rewards"]))

    summary[method_name] = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_p50": float(np.percentile(rewards, 50)),
        "reward_p95": float(np.percentile(rewards, 95)),
        "latency_mean": float(np.mean(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "bandwidth_mean": float(np.mean(bandwidths)),
        "latency_probes": int(method_results["latency_probes"]),
        "bandwidth_probes": int(method_results["bandwidth_probes"]),
        "total_probes": int(
            method_results["latency_probes"] + method_results["bandwidth_probes"]
        ),
        "total_probe_time_ms": probe_time,
        "avg_probe_time_per_selection": probe_time / n_selections,
        "avg_selection_time_ms": float(np.mean(selection_times)),
        "n_selections": int(n_selections),
    }

    s = summary[method_name]
    print(
        f"\n{method_name.upper()}:"
        f"\n  Reward: {s['reward_mean']:.3f} ± {s['reward_std']:.3f}"
        f"\n  Latency (ms): {s['latency_mean']:.1f} (p95: {s['latency_p95']:.1f})"
        f"\n  Bandwidth (Mbps): {s['bandwidth_mean']:.1f}"
        f"\n  Total probes: {s['total_probes']} "
        f"({s['latency_probes']} latency, {s['bandwidth_probes']} bandwidth)"
        f"\n  Avg probe overhead per selection: {s['avg_probe_time_per_selection']:.1f} ms"
        f"\n  Avg selection time: {s['avg_selection_time_ms']:.3f} ms"
    )


# Probe reduction vs. baseline mean (DQN highlight).
probe_reduction = 0.0
time_reduction = 0.0
if "dqn" in summary:
    baseline_avg_probes = float(
        np.mean(
            [s["total_probes"] for k, s in summary.items() if k != "dqn"]
            or [0.0]
        )
    )
    if baseline_avg_probes:
        probe_reduction = (
            (baseline_avg_probes - summary["dqn"]["total_probes"])
            / baseline_avg_probes
            * 100.0
        )
    baseline_avg_time = float(
        np.mean(
            [s["total_probe_time_ms"] for k, s in summary.items() if k != "dqn"]
            or [0.0]
        )
    )
    if baseline_avg_time:
        time_reduction = (
            (baseline_avg_time - summary["dqn"]["total_probe_time_ms"])
            / baseline_avg_time
            * 100.0
        )

    print("\n" + "=" * 60)
    print("DQN PROBE REDUCTION vs. baseline mean:")
    print(f"  Probe count reduction:  {probe_reduction:.1f}%")
    print(f"  Probe time reduction:   {time_reduction:.1f}%")


with open(run_path / "evaluation_results.json", "w") as f:
    json.dump(
        {
            "summary": summary,
            "num_eval_pairs": len(EVAL_PAIRS),
            "num_eval_hours": len(EVAL_HOURS),
            "num_eval_selections": int(len(EVAL_HOURS) * len(EVAL_PAIRS)),
            "num_paths_action_dim": _action_dim,
            "probe_reduction_percent": probe_reduction,
            "time_reduction_percent": time_reduction,
        },
        f,
        indent=2,
    )

print(f"\nResults saved to: {run_path / 'evaluation_results.json'}")
print("\nEvaluation complete!")
