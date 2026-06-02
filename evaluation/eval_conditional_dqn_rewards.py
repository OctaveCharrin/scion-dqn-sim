#!/usr/bin/env python3
"""Evaluate conditional DQN under multiple reward-weight profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.pipeline.run_dirs import resolve_run_dir
from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import ConditionalPathScoringDQNAgent
from src.rl.reward_profiles import REWARD_PROFILES, RewardProfile
from src.simulation.evaluation_env import (
    CONDITIONAL_SCORING_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    RewardWeights,
)
from src.simulation.run_context import load_run_context, make_env

EVAL_HOURS = list(range(14 * 24, 28 * 24))


def _load_agent(run_path: Path) -> ConditionalPathScoringDQNAgent:
    ckpt_path = run_path / "dqn_conditional_scoring_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint {ckpt_path}; run 04_train_conditional_dqn.py first."
        )
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = ckpt.get("config") or EnhancedDQNConfig()
    agent = ConditionalPathScoringDQNAgent(
        global_dim=int(ckpt.get("global_dim", CONDITIONAL_SCORING_GLOBAL_DIM)),
        path_dim=int(ckpt.get("path_dim", PATH_FEATURE_DIM)),
        config=cfg,
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    if "target_network" in ckpt:
        agent.target_network.load_state_dict(ckpt["target_network"])
    agent.epsilon = 0.0
    return agent


def _eval_profile(
    env,
    agent: ConditionalPathScoringDQNAgent,
    profile: RewardProfile,
    eval_pairs: List[Tuple[int, int]],
) -> Dict[str, Any]:
    env.reward_weights = profile.weights
    rewards: List[float] = []
    bandwidths: List[float] = []
    latencies: List[float] = []
    losses: List[float] = []

    for hour_idx in EVAL_HOURS:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            state = env.observe_scoring_conditional()
            if state["paths"].shape[0] == 0:
                continue
            action = int(agent.act(state, evaluate=True))
            if action >= len(env.available_paths):
                action = 0
            reward, _, info = env.apply_action(action, probe="full")
            pm = info["path_metrics"]
            rewards.append(float(reward))
            bw = pm.get("bandwidth_mbps")
            bandwidths.append(float(bw) if bw is not None else 0.0)
            latencies.append(float(pm.get("latency_ms", 50.0)))
            losses.append(float(pm.get("loss_rate", 0.0)))

    n = len(rewards)
    return {
        "profile": profile.to_dict(),
        "n_selections": n,
        "reward_mean": float(np.mean(rewards)) if n else 0.0,
        "reward_std": float(np.std(rewards)) if n else 0.0,
        "bandwidth_mean_mbps": float(np.mean(bandwidths)) if n else 0.0,
        "latency_mean_ms": float(np.mean(latencies)) if n else 0.0,
        "loss_mean": float(np.mean(losses)) if n else 0.0,
    }


def _eval_oracle_greedy(
    env,
    profile: RewardProfile,
    eval_pairs: List[Tuple[int, int]],
    goodput_cap: float,
) -> Dict[str, Any]:
    """Upper bound: probe all paths and pick best reward under this profile."""
    env.reward_weights = profile.weights
    rewards: List[float] = []
    bandwidths: List[float] = []
    latencies: List[float] = []

    for hour_idx in EVAL_HOURS:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            n_paths = len(env.available_paths)
            if n_paths == 0:
                continue
            best_r = -1e9
            best_pm: Dict[str, Any] = {}
            for path_idx in range(n_paths):
                env.probe_path_full(path_idx)
                pm = env.probed_path_metrics[path_idx]
                r = env.compute_reward(
                    pm,
                    max_possible_bw=goodput_cap,
                    probe_cost_ms=env.last_probe_cost_ms,
                    num_probes_in_step=n_paths,
                )
                if r > best_r:
                    best_r = r
                    best_pm = pm
            rewards.append(float(best_r))
            bw = best_pm.get("bandwidth_mbps")
            bandwidths.append(float(bw) if bw is not None else 0.0)
            latencies.append(float(best_pm.get("latency_ms", 50.0)))

    n = len(rewards)
    return {
        "method": "oracle_greedy_reward",
        "profile": profile.name,
        "n_selections": n,
        "reward_mean": float(np.mean(rewards)) if n else 0.0,
        "bandwidth_mean_mbps": float(np.mean(bandwidths)) if n else 0.0,
        "latency_mean_ms": float(np.mean(latencies)) if n else 0.0,
    }


def main() -> None:
    run_path = Path(resolve_run_dir())
    topology_data, path_store, link_states, pair_pool, goodput_cap = load_run_context(
        run_path
    )
    with open(run_path / "selected_pair.json") as f:
        selected_pair = json.load(f)
    eval_pairs = pair_pool[: min(len(pair_pool), 32)] or [
        (int(selected_pair["source_as"]), int(selected_pair["destination_as"]))
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
    agent = _load_agent(run_path)

    print(f"\nConditional DQN multi-reward evaluation on {run_path}")
    print(f"  eval pairs: {len(eval_pairs)}, hours: {len(EVAL_HOURS)}")

    results: Dict[str, Any] = {
        "run_dir": str(run_path),
        "eval_pairs": [list(p) for p in eval_pairs],
        "profiles": [],
        "oracle_by_profile": [],
    }

    for profile in tqdm(REWARD_PROFILES, desc="profiles"):
        row = _eval_profile(env, agent, profile, eval_pairs)
        results["profiles"].append(row)
        oracle = _eval_oracle_greedy(env, profile, eval_pairs, goodput_cap)
        results["oracle_by_profile"].append(oracle)

    print("\n" + "=" * 72)
    print(f"{'Profile':<18} {'Reward':>10} {'BW Mbps':>10} {'Lat ms':>10} {'Oracle R':>10}")
    print("-" * 72)
    for row, oracle in zip(results["profiles"], results["oracle_by_profile"]):
        name = row["profile"]["name"]
        print(
            f"{name:<18} "
            f"{row['reward_mean']:>10.3f} "
            f"{row['bandwidth_mean_mbps']:>10.1f} "
            f"{row['latency_mean_ms']:>10.1f} "
            f"{oracle['reward_mean']:>10.3f}"
        )

    out_path = run_path / "conditional_dqn_reward_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time() - t0:.1f}s")
