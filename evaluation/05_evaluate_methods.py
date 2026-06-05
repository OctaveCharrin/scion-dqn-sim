#!/usr/bin/env python3
"""Evaluate path-selection methods on the last 14 days (multi-pair).

Uses the unified :class:`~src.simulation.evaluation_env.EvaluationPathSelectionEnv`
for observations, probing, and reward computation.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.evaluation_env import (
    CONDITIONAL_SCORING_GLOBAL_DIM,
    FLAT_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    SCORING_GLOBAL_DIM,
    RewardWeights,
)
from src.rl.dqn_agent_scoring_conditional import load_conditional_scoring_agent
from src.simulation.run_context import (
    compute_action_dim,
    load_run_context,
    make_env,
)

from src.baselines.ecmp import ECMPSelector
from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.random_selection import RandomSelector
from src.baselines.scion_default import SCIONDefaultSelector
from src.baselines.shortest_path import ShortestPathSelector
from src.baselines.widest_path import WidestPathSelector
from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
from src.rl.dqn_agent_scoring_simple import SimplePathScoringDQNAgent
from src.rl.dqn_agent_simple import SimpleDQNAgent

run_path = Path(resolve_run_dir())

topology_data, path_store, link_states, pair_pool, _goodput_cap = load_run_context(run_path)
with open(run_path / "selected_pair.json", "r") as f:
    selected_pair = json.load(f)

pair_pool: List[Tuple[int, int]] = pair_pool or [
    (int(selected_pair["source_as"]), int(selected_pair["destination_as"]))
]
EVAL_PAIRS = pair_pool[: min(len(pair_pool), 32)]
EVAL_HOURS = list(range(14 * 24, 28 * 24))
action_dim = compute_action_dim(path_store, selected_pair)

print(f"\nEvaluating across {len(EVAL_PAIRS)} pair(s) (of {len(pair_pool)} in pool)")
print(f"Evaluation horizon: {len(EVAL_HOURS)} hours x {len(EVAL_PAIRS)} pairs")

reward_weights = RewardWeights()
env = make_env(
    topology_data,
    path_store,
    link_states,
    EVAL_PAIRS,
    episode_length=1,
    rng_seed=7,
    reward_weights=reward_weights,
)


def _load_checkpoint(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# --- Load trained agents -------------------------------------------------------

dqn_agent: Optional[EnhancedDQNAgent] = None
simple_dqn_agent: Optional[SimpleDQNAgent] = None
scoring_simple_dqn_agent: Optional[SimplePathScoringDQNAgent] = None
scoring_enhanced_dqn_agent: Optional[EnhancedPathScoringDQNAgent] = None
conditional_dqn_agent: Optional[EnhancedPathScoringDQNAgent] = None

_model_path = run_path / "dqn_model.pth"
if _model_path.is_file():
    model_checkpoint = _load_checkpoint(_model_path)
    if model_checkpoint:
        config: EnhancedDQNConfig = model_checkpoint["config"]
        dqn_agent = EnhancedDQNAgent(
            int(model_checkpoint.get("state_dim", FLAT_GLOBAL_DIM)),
            int(model_checkpoint.get("action_dim", action_dim)),
            config,
        )
        dqn_agent.q_network.load_state_dict(model_checkpoint["q_network"])
        if "target_network" in model_checkpoint:
            dqn_agent.target_network.load_state_dict(model_checkpoint["target_network"])
        dqn_agent.epsilon = 0.0
        rw = model_checkpoint.get("reward_weights")
        if rw:
            reward_weights = RewardWeights.from_mapping(rw)

_simple_path = run_path / "dqn_simple_model.pth"
if _simple_path.is_file():
    ckpt = _load_checkpoint(_simple_path)
    if ckpt:
        simple_dqn_agent = SimpleDQNAgent(
            state_dim=int(ckpt.get("state_dim", FLAT_GLOBAL_DIM)),
            action_dim=int(ckpt.get("action_dim", action_dim)),
            learning_rate=1e-3,
        )
        simple_dqn_agent.q_network.load_state_dict(ckpt["q_network"])
        if "target_network" in ckpt:
            simple_dqn_agent.target_network.load_state_dict(ckpt["target_network"])
        simple_dqn_agent.epsilon = 0.0

_ckpt = _load_checkpoint(run_path / "dqn_scoring_simple_model.pth")
if _ckpt:
    scoring_simple_dqn_agent = SimplePathScoringDQNAgent(
        global_dim=int(_ckpt.get("global_dim", SCORING_GLOBAL_DIM)),
        path_dim=int(_ckpt.get("path_dim", PATH_FEATURE_DIM)),
        hidden_dim=int(_ckpt.get("hidden_dim", 128)),
    )
    scoring_simple_dqn_agent.q_network.load_state_dict(_ckpt["q_network"])
    if "target_network" in _ckpt:
        scoring_simple_dqn_agent.target_network.load_state_dict(_ckpt["target_network"])
    scoring_simple_dqn_agent.epsilon = 0.0
    if _ckpt.get("reward_weights"):
        reward_weights = RewardWeights.from_mapping(_ckpt["reward_weights"])

_enh_path = run_path / "dqn_scoring_enhanced_model.pth"
_enh_ckpt = _load_checkpoint(_enh_path)
if _enh_ckpt:
    cfg = _enh_ckpt.get("config") or EnhancedDQNConfig()
    scoring_enhanced_dqn_agent = EnhancedPathScoringDQNAgent(
        global_dim=int(_enh_ckpt.get("global_dim", SCORING_GLOBAL_DIM)),
        path_dim=int(_enh_ckpt.get("path_dim", PATH_FEATURE_DIM)),
        config=cfg,
    )
    scoring_enhanced_dqn_agent.q_network.load_state_dict(_enh_ckpt["q_network"])
    if "target_network" in _enh_ckpt:
        scoring_enhanced_dqn_agent.target_network.load_state_dict(_enh_ckpt["target_network"])
    scoring_enhanced_dqn_agent.epsilon = 0.0

_cond_path = run_path / "dqn_conditional_scoring_model.pth"
_cond_ckpt = _load_checkpoint(_cond_path)
if _cond_ckpt:
    conditional_dqn_agent = load_conditional_scoring_agent(_cond_ckpt)

env.reward_weights = reward_weights

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

_nn_methods: List[Tuple[str, object]] = []
if dqn_agent is not None:
    _nn_methods.append(("dqn", dqn_agent))
if simple_dqn_agent is not None:
    _nn_methods.append(("simple_dqn", simple_dqn_agent))
if scoring_simple_dqn_agent is not None:
    _nn_methods.append(("scoring_simple_dqn", scoring_simple_dqn_agent))
if scoring_enhanced_dqn_agent is not None:
    _nn_methods.append(("scoring_enhanced_dqn", scoring_enhanced_dqn_agent))
if conditional_dqn_agent is not None:
    _nn_methods.append(("conditional_dqn", conditional_dqn_agent))


def _record_step(method_results: Dict, info: Dict, reward: float, selection_time_ms: float) -> None:
    pm = info["path_metrics"]
    method_results["rewards"].append(reward)
    method_results["latencies"].append(float(pm.get("latency_ms", 50.0)))
    bw_val = pm.get("bandwidth_mbps")
    method_results["bandwidths"].append(float(bw_val) if bw_val is not None else 0.0)
    method_results["losses"].append(float(pm.get("loss_rate", 0.0)))
    method_results["selection_times_ms"].append(selection_time_ms)


print("\nEvaluating methods...")

for method_name, method in _nn_methods + list(baseline_methods.items()):
    print(f"\n--- {method_name} ---")
    method_results = results[method_name]
    pbar = tqdm(total=len(EVAL_HOURS) * len(EVAL_PAIRS), desc=method_name, ncols=80)

    for hour_idx in EVAL_HOURS:
        for pair in EVAL_PAIRS:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            paths = env.available_paths
            if not paths:
                pbar.update(1)
                continue

            t0 = time.time()
            step_probe_cost = 0.0

            if method_name == "dqn":
                state = env.observe_flat()
                mask = env.action_mask(action_dim)
                action = int(dqn_agent.act(state, action_mask=mask))
                if action >= len(paths):
                    valid = np.where(mask)[0]
                    action = int(valid[0]) if len(valid) > 0 else 0
                reward, _, info = env.apply_action(action, probe="full")
                method_results["latency_probes"] += 1
                method_results["bandwidth_probes"] += 1
                method_results["total_probe_time_ms"] += info["step_probe_cost_ms"]

            elif method_name == "simple_dqn":
                state = env.observe_flat()
                action = int(simple_dqn_agent.act(state))
                if action >= len(paths):
                    action = 0
                reward, _, info = env.apply_action(action, probe="full")
                method_results["latency_probes"] += 1
                method_results["bandwidth_probes"] += 1
                method_results["total_probe_time_ms"] += info["step_probe_cost_ms"]

            elif method_name in ("scoring_simple_dqn", "scoring_enhanced_dqn"):
                state_d = env.observe_scoring()
                if state_d["paths"].shape[0] == 0:
                    pbar.update(1)
                    continue
                action = int(method.act(state_d, evaluate=True))
                if action >= len(paths):
                    action = 0
                reward, _, info = env.apply_action(action, probe="full")
                method_results["latency_probes"] += 1
                method_results["bandwidth_probes"] += 1
                method_results["total_probe_time_ms"] += info["step_probe_cost_ms"]

            elif method_name == "conditional_dqn":
                state_d = env.observe_scoring_conditional()
                if state_d["paths"].shape[0] == 0:
                    pbar.update(1)
                    continue
                action = int(conditional_dqn_agent.act(state_d, evaluate=True))
                if action >= len(paths):
                    action = 0
                reward, _, info = env.apply_action(action, probe="full")
                method_results["latency_probes"] += 1
                method_results["bandwidth_probes"] += 1
                method_results["total_probe_time_ms"] += info["step_probe_cost_ms"]

            else:
                path_metrics_list: List[Dict] = []
                n_probes_step = 0
                for path_idx in range(len(paths)):
                    if method_name in ("widest_path", "ecmp"):
                        m = env.probe_path_full(path_idx)
                        method_results["latency_probes"] += 1
                        method_results["bandwidth_probes"] += 1
                    else:
                        m = env.probe_path_latency(path_idx)
                        method_results["latency_probes"] += 1
                    step_probe_cost += env.last_probe_cost_ms
                    n_probes_step += 1
                    path_metrics_list.append(m)

                flow_stub = {"src": int(pair[0]), "dst": int(pair[1])}
                state_stub = np.zeros(1, dtype=np.float32)
                if method_name == "random":
                    action = int(np.random.choice(len(paths)))
                else:
                    action = int(
                        method.select_path(paths, path_metrics_list, flow_stub, state_stub)
                    )
                _, reward, _, info = env.step(
                    action,
                    step_probe_cost_ms=step_probe_cost,
                    num_probes_in_step=n_probes_step,
                )
                method_results["total_probe_time_ms"] += step_probe_cost

            _record_step(method_results, info, reward, (time.time() - t0) * 1000.0)
            pbar.update(1)
    pbar.close()


# --- Summary -------------------------------------------------------------------

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
        f"\n  Total probes: {s['total_probes']}"
        f"\n  Avg probe overhead per selection: {s['avg_probe_time_per_selection']:.1f} ms"
    )

probe_reduction = 0.0
time_reduction = 0.0
if "dqn" in summary:
    baseline_avg_probes = float(
        np.mean([s["total_probes"] for k, s in summary.items() if k != "dqn"] or [0.0])
    )
    if baseline_avg_probes:
        probe_reduction = (
            (baseline_avg_probes - summary["dqn"]["total_probes"]) / baseline_avg_probes * 100.0
        )
    baseline_avg_time = float(
        np.mean(
            [s["total_probe_time_ms"] for k, s in summary.items() if k != "dqn"] or [0.0]
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
            "num_paths_action_dim": action_dim,
            "probe_reduction_percent": probe_reduction,
            "time_reduction_percent": time_reduction,
            "reward_weights": reward_weights.to_dict(),
        },
        f,
        indent=2,
    )

print(f"\nResults saved to: {run_path / 'evaluation_results.json'}")
print("\nEvaluation complete!")
