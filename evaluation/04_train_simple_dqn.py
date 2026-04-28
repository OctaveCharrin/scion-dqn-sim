#!/usr/bin/env python3
"""Train the simple DQN agent (single-pair, paper-style).

Single-pair counterpart of ``04_train_dqn.py`` for the simpler MLP DQN that
doesn't support action masking. It uses the same stateful environment and the
same reward shape so the two variants stay comparable.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from _common import resolve_run_dir, topology_dir

from src.rl.dqn_agent_simple import SimpleDQNAgent
from src.simulation.evaluation_env import EvaluationPathSelectionEnv

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

src_as = int(selected_pair["source_as"])
dst_as = int(selected_pair["destination_as"])
num_paths = int(selected_pair["num_paths"])
print(f"\nTraining Simple DQN for AS {src_as} -> AS {dst_as} ({num_paths} paths)")


# -----------------------------------------------------------------------------
# Goodput cap (P95 of static min bandwidths for this pair)
# -----------------------------------------------------------------------------

_pair_paths = path_store.find_paths(src_as, dst_as) or []
_min_bws = [
    float((p.get("static_metrics") or {}).get("min_bandwidth", 100.0))
    for p in _pair_paths
]
GOODPUT_CAP_MBPS = max(50.0, float(np.percentile(_min_bws, 95)) if _min_bws else 100.0)
print(f"  Goodput cap: {GOODPUT_CAP_MBPS:.0f} Mbps")


# -----------------------------------------------------------------------------
# Environment + agent
# -----------------------------------------------------------------------------

EPISODE_LENGTH = 24
env = EvaluationPathSelectionEnv(
    topology_data=topology_data,
    path_store=path_store,
    link_states=link_states,
    latency_probe_cost_ms=10.0,
    bandwidth_probe_cost_ms=100.0,
    pair_pool=[(src_as, dst_as)],
    episode_length=EPISODE_LENGTH,
    rng_seed=42,
)

state_dim = 5
agent = SimpleDQNAgent(
    state_dim=state_dim,
    action_dim=num_paths,
    learning_rate=1e-3,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10_000,
    batch_size=32,
    target_update_every=100,
    min_buffer_size=500,
)

W1, W2, W3, W4 = 0.7, 0.3, 0.5, 0.5
print("\nDQN configuration:")
print(f"  State dimensions: {state_dim}")
print(f"  Action space: {num_paths} paths")
print(f"  Reward weights: w1={W1}, w2={W2}, w3={W3}, w4={W4}")
print(f"  Episode length (hours): {EPISODE_LENGTH}")
print(f"  Device: {agent.device}")


# -----------------------------------------------------------------------------
# State + reward (must match 04_train_dqn.py)
# -----------------------------------------------------------------------------


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
        max(0.0, min(1.0, 1.0 - (W3 * loss + W4 * lat)))
        for loss, lat in zip(losses, lats)
    ]
    f2 = float(np.mean(utils)) if utils else 0.0
    f3 = float(np.mean(trusts)) if trusts else 0.0
    f4 = float(np.mean([1.0 if u > 0.7 else 0.0 for u in utils])) if utils else 0.0
    return np.array([f0, f1, f2, f3, f4], dtype=np.float32)


def _reward_from_metrics(path_metrics: Dict, cap_mbps: float) -> float:
    bw = float(path_metrics.get("bandwidth_mbps", 0.0))
    goodput = max(0.0, min(bw / cap_mbps, 1.0))
    loss = float(path_metrics.get("loss_rate", 0.0))
    delay = min(100.0, float(path_metrics.get("latency_ms", 50.0))) / 100.0
    trust = max(0.0, min(1.0, 1.0 - (W3 * loss + W4 * delay)))
    return float(2.0 * (W1 * goodput + W2 * trust) - 1.0)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

TRAINING_HOURS = list(range(14 * 24))
NUM_EPISODES = max(200, len(traffic_flows) // EPISODE_LENGTH)

import os as _os

_env_eps = _os.environ.get("DQN_TRAIN_EPISODES", "").strip()
if _env_eps.isdigit():
    NUM_EPISODES = int(_env_eps)

print(
    f"\nTraining: {NUM_EPISODES} episodes x {EPISODE_LENGTH} hours each "
    f"(~{NUM_EPISODES * EPISODE_LENGTH} steps)..."
)

import random

start_rng = random.Random(123)
episode_rewards: List[float] = []
episode_probes: List[int] = []
losses: List[float] = []

for ep in tqdm(range(NUM_EPISODES), desc="Episodes"):
    start_hour = start_rng.choice(TRAINING_HOURS)
    env.reset(source_as=src_as, dest_as=dst_as, hour_idx=start_hour)
    state = _aggregate_state(env, env.hour_idx)
    ep_reward = 0.0
    for step in range(EPISODE_LENGTH):
        action = int(agent.act(state))
        if action >= len(env.available_paths):
            action = 0
        _, _, done, info = env.step(action)
        next_state = _aggregate_state(env, env.hour_idx)
        reward = _reward_from_metrics(info["path_metrics"], GOODPUT_CAP_MBPS)
        ep_reward += reward
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        if loss is not None:
            losses.append(float(loss))
        state = next_state
        if done:
            break
    agent.episodes += 1
    agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
    episode_rewards.append(ep_reward / max(1, EPISODE_LENGTH))
    episode_probes.append(int(env.num_latency_probes + env.num_bandwidth_probes))


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------

model_file = run_path / "dqn_simple_model.pth"
torch.save(
    {
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "steps": agent.steps,
        "episodes": agent.episodes,
        "state_dim": state_dim,
        "action_dim": num_paths,
        "goodput_cap_mbps": GOODPUT_CAP_MBPS,
        "reward_weights": {"w1": W1, "w2": W2, "w3": W3, "w4": W4},
    },
    model_file,
)
print(f"\nSimple model saved to: {model_file}")

training_stats = {
    "num_episodes": NUM_EPISODES,
    "episode_length_hours": EPISODE_LENGTH,
    "episode_rewards": episode_rewards,
    "episode_probes": episode_probes,
    "losses": losses,
    "final_epsilon": agent.epsilon,
    "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    "avg_probes_per_episode": float(np.mean(episode_probes))
    if episode_probes
    else 0.0,
    "reward_weights": {"w1": W1, "w2": W2, "w3": W3, "w4": W4},
    "goodput_cap_mbps": GOODPUT_CAP_MBPS,
}
with open(run_path / "dqn_simple_training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=4)
print(f"Simple training stats saved to: {run_path / 'dqn_simple_training_stats.json'}")
