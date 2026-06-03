#!/usr/bin/env python3
"""Compare path-selection methods under multiple reward-weight profiles (multi-pair)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.widest_path import WidestPathSelector
from src.pipeline.run_dirs import resolve_run_dir
from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import ConditionalPathScoringDQNAgent
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
from src.rl.dqn_agent_scoring_simple import SimplePathScoringDQNAgent
from src.rl.dqn_agent_simple import SimpleDQNAgent
from src.rl.reward_profiles import REWARD_PROFILES, RewardProfile
from src.simulation.evaluation_env import (
    CONDITIONAL_SCORING_GLOBAL_DIM,
    FLAT_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    SCORING_GLOBAL_DIM,
    RewardWeights,
)
from src.simulation.run_context import compute_action_dim, load_run_context, make_env

EVAL_HOURS = list(range(14 * 24, 28 * 24))
MAX_EVAL_PAIRS = 32


def _load_checkpoint(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_agents(run_path: Path, action_dim: int) -> Dict[str, Any]:
    agents: Dict[str, Any] = {}

    ckpt = _load_checkpoint(run_path / "dqn_conditional_scoring_model.pth")
    if ckpt:
        cfg = ckpt.get("config") or EnhancedDQNConfig()
        agent = ConditionalPathScoringDQNAgent(
            global_dim=int(ckpt.get("global_dim", CONDITIONAL_SCORING_GLOBAL_DIM)),
            path_dim=int(ckpt.get("path_dim", PATH_FEATURE_DIM)),
            config=cfg,
        )
        agent.q_network.load_state_dict(ckpt["q_network"])
        agent.epsilon = 0.0
        agents["conditional_dqn"] = agent

    ckpt = _load_checkpoint(run_path / "dqn_scoring_enhanced_model.pth")
    if ckpt:
        gdim = int(ckpt.get("global_dim", SCORING_GLOBAL_DIM))
        pdim = int(ckpt.get("path_dim", PATH_FEATURE_DIM))
        cfg = ckpt.get("config") or EnhancedDQNConfig()
        agent = EnhancedPathScoringDQNAgent(global_dim=gdim, path_dim=pdim, config=cfg)
        agent.q_network.load_state_dict(ckpt["q_network"])
        agent.epsilon = 0.0
        agents["scoring_enhanced_dqn"] = agent

    ckpt = _load_checkpoint(run_path / "dqn_scoring_simple_model.pth")
    if ckpt:
        agent = SimplePathScoringDQNAgent(
            global_dim=int(ckpt.get("global_dim", SCORING_GLOBAL_DIM)),
            path_dim=int(ckpt.get("path_dim", PATH_FEATURE_DIM)),
            hidden_dim=int(ckpt.get("hidden_dim", 128)),
        )
        agent.q_network.load_state_dict(ckpt["q_network"])
        agent.epsilon = 0.0
        agents["scoring_simple_dqn"] = agent

    ckpt = _load_checkpoint(run_path / "dqn_model.pth")
    if ckpt:
        cfg = ckpt.get("config") or EnhancedDQNConfig()
        agent = EnhancedDQNAgent(
            int(ckpt.get("state_dim", FLAT_GLOBAL_DIM)),
            int(ckpt.get("action_dim", action_dim)),
            cfg,
        )
        agent.q_network.load_state_dict(ckpt["q_network"])
        agent.epsilon = 0.0
        agents["dqn"] = agent

    ckpt = _load_checkpoint(run_path / "dqn_simple_model.pth")
    if ckpt:
        agent = SimpleDQNAgent(
            state_dim=int(ckpt.get("state_dim", FLAT_GLOBAL_DIM)),
            action_dim=int(ckpt.get("action_dim", action_dim)),
        )
        agent.q_network.load_state_dict(ckpt["q_network"])
        agent.epsilon = 0.0
        agents["simple_dqn"] = agent

    agents["widest_path"] = WidestPathSelector()
    agents["lowest_latency"] = LowestLatencySelector()
    return agents


def _eval_method_profile(
    env,
    method: str,
    agent: Any,
    profile: RewardProfile,
    eval_pairs: List[Tuple[int, int]],
    action_dim: int,
    _goodput_cap: float,
) -> Dict[str, Any]:
    """Evaluate one method under one reward profile."""
    env.reward_weights = profile.weights
    rewards: List[float] = []
    bandwidths: List[float] = []
    latencies: List[float] = []
    losses: List[float] = []
    probe_ms: List[float] = []

    for hour_idx in EVAL_HOURS:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            n_paths = len(env.available_paths)
            if n_paths == 0:
                continue

            step_probe_cost = 0.0
            if method in ("widest_path", "lowest_latency"):
                path_metrics_list: List[Dict] = []
                n_probes = 0
                for path_idx in range(n_paths):
                    if method == "widest_path":
                        m = env.probe_path_full(path_idx)
                    else:
                        m = env.probe_path_latency(path_idx)
                    step_probe_cost += env.last_probe_cost_ms
                    n_probes += 1
                    path_metrics_list.append(m)
                flow_stub = {"src": int(pair[0]), "dst": int(pair[1])}
                action = int(
                    agent.select_path(
                        env.available_paths,
                        path_metrics_list,
                        flow_stub,
                        np.zeros(1, dtype=np.float32),
                    )
                )
                _, reward, _, info = env.step(
                    action,
                    step_probe_cost_ms=step_probe_cost,
                    num_probes_in_step=n_probes,
                )
            elif method == "conditional_dqn":
                state = env.observe_scoring_conditional()
                if state["paths"].shape[0] == 0:
                    continue
                action = int(agent.act(state, evaluate=True))
                if action >= n_paths:
                    action = 0
                reward, _done, info = env.apply_action(action, probe="full")
            elif method in ("scoring_enhanced_dqn", "scoring_simple_dqn"):
                state = env.observe_scoring()
                if state["paths"].shape[0] == 0:
                    continue
                action = int(agent.act(state, evaluate=True))
                if action >= n_paths:
                    action = 0
                reward, _done, info = env.apply_action(action, probe="full")
            elif method == "dqn":
                state = env.observe_flat()
                mask = env.action_mask(action_dim)
                action = int(agent.act(state, action_mask=mask))
                if action >= n_paths:
                    valid = np.where(mask)[0]
                    action = int(valid[0]) if len(valid) else 0
                reward, _done, info = env.apply_action(action, probe="full")
            else:
                state = env.observe_flat()
                action = int(agent.act(state))
                if action >= n_paths:
                    action = 0
                reward, _done, info = env.apply_action(action, probe="full")

            pm = info["path_metrics"]
            rewards.append(float(reward))
            bw = pm.get("bandwidth_mbps")
            bandwidths.append(float(bw) if bw is not None else 0.0)
            latencies.append(float(pm.get("latency_ms", 50.0)))
            losses.append(float(pm.get("loss_rate", 0.0)))
            probe_ms.append(float(info.get("step_probe_cost_ms", env.last_probe_cost_ms)))

    n = len(rewards)
    return {
        "method": method,
        "profile": profile.name,
        "weights": profile.weights.to_dict(),
        "n_selections": n,
        "reward_mean": float(np.mean(rewards)) if n else 0.0,
        "reward_std": float(np.std(rewards)) if n else 0.0,
        "bandwidth_mean_mbps": float(np.mean(bandwidths)) if n else 0.0,
        "latency_mean_ms": float(np.mean(latencies)) if n else 0.0,
        "loss_mean": float(np.mean(losses)) if n else 0.0,
        "probe_cost_mean_ms": float(np.mean(probe_ms)) if n else 0.0,
    }


def _print_tables(results: List[Dict[str, Any]], profiles: List[str], methods: List[str]) -> None:
    print("\n" + "=" * 88)
    print("MEAN REWARD BY METHOD AND REWARD PROFILE")
    print("=" * 88)
    header = f"{'Method':<24}" + "".join(f"{p[:10]:>11}" for p in profiles)
    print(header)
    print("-" * len(header))
    lookup = {(r["method"], r["profile"]): r["reward_mean"] for r in results}
    for method in methods:
        row = f"{method:<24}"
        for profile in profiles:
            row += f"{lookup.get((method, profile), float('nan')):>11.3f}"
        print(row)

    print("\n" + "=" * 88)
    print("MEAN BANDWIDTH (Mbps) BY METHOD AND PROFILE")
    print("=" * 88)
    print(header)
    print("-" * len(header))
    lookup_bw = {(r["method"], r["profile"]): r["bandwidth_mean_mbps"] for r in results}
    for method in methods:
        row = f"{method:<24}"
        for profile in profiles:
            row += f"{lookup_bw.get((method, profile), float('nan')):>11.1f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=MAX_EVAL_PAIRS,
        help=f"Cap eval pairs (default {MAX_EVAL_PAIRS}).",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    topology_data, path_store, link_states, pair_pool, goodput_cap = load_run_context(
        run_path
    )
    with open(run_path / "selected_pair.json") as f:
        selected_pair = json.load(f)
    eval_pairs = pair_pool[: min(len(pair_pool), args.max_pairs)] or [
        (int(selected_pair["source_as"]), int(selected_pair["destination_as"]))
    ]
    action_dim = compute_action_dim(path_store, selected_pair)

    agents = _load_agents(run_path, action_dim)
    if not agents:
        raise FileNotFoundError(f"No model checkpoints found under {run_path}")

    method_order = [
        m
        for m in (
            "conditional_dqn",
            "scoring_enhanced_dqn",
            "scoring_simple_dqn",
            "dqn",
            "simple_dqn",
            "widest_path",
            "lowest_latency",
        )
        if m in agents
    ]

    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=RewardWeights(),
    )

    print(f"\nMulti-reward comparison on {run_path}")
    print(f"  eval pairs: {len(eval_pairs)} / {len(pair_pool)} in pool")
    print(f"  hours: {len(EVAL_HOURS)}, methods: {', '.join(method_order)}")
    print(f"  profiles: {', '.join(p.name for p in REWARD_PROFILES)}")

    results: List[Dict[str, Any]] = []
    total = len(REWARD_PROFILES) * len(method_order)
    pbar = tqdm(total=total, desc="method×profile", ncols=88)

    for profile in REWARD_PROFILES:
        for method in method_order:
            row = _eval_method_profile(
                env, method, agents[method], profile, eval_pairs, action_dim, goodput_cap
            )
            results.append(row)
            pbar.update(1)
    pbar.close()

    profile_names = [p.name for p in REWARD_PROFILES]
    _print_tables(results, profile_names, method_order)

    out = {
        "run_dir": str(run_path),
        "eval_pairs": [list(p) for p in eval_pairs],
        "pair_pool_size": len(pair_pool),
        "n_eval_hours": len(EVAL_HOURS),
        "methods": method_order,
        "profiles": profile_names,
        "results": results,
    }
    out_path = run_path / "multi_reward_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time() - t0:.1f}s")
