#!/usr/bin/env python3
"""Train a DQN agent for SCION path selection.

Notable behavior changes vs. the original:

* **Multi-pair training.** The agent is trained against every AS pair in
  ``selected_pair["pair_pool"]`` (falling back to the legacy single-pair
  scheme when no pool is present). The action dimension is the maximum
  ``num_paths`` over all pairs and an action mask hides invalid actions per
  pair, so the policy generalizes across SCION-shaped destinations.
* **Stateful episodes.** ``EvaluationPathSelectionEnv`` now advances ``hour_idx``
  step-by-step and ``done`` fires after ``episode_length`` steps, so γ in the
  Bellman bootstrap actually contributes signal.
* **Real link-derived state.** The agent observes per-pair, per-hour aggregate
  utilization and link-trust statistics computed from the current pair's path
  metrics — no more hand-tuned ``avg_*`` placeholders.
* **Reward.** The composite goodput / link-trust reward keeps the same
  ``r = 2*(w1*G + w2*T) - 1`` shape, but with an adaptive goodput cap derived
  from the topology's static bandwidths so a saturated reward stops being the
  default. ``w3`` / ``w4`` continue to weight loss vs. delay inside the trust
  term.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from _common import resolve_run_dir, topology_dir

from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
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

pair_pool: List[Tuple[int, int]] = [
    (int(p[0]), int(p[1]))
    for p in selected_pair.get("pair_pool", [[src_as, dst_as]])
]
if not pair_pool:
    pair_pool = [(src_as, dst_as)]


def _num_paths_for(pair: Tuple[int, int]) -> int:
    return len(path_store.find_paths(int(pair[0]), int(pair[1])) or [])


pair_paths_count: Dict[Tuple[int, int], int] = {p: _num_paths_for(p) for p in pair_pool}
action_dim = int(
    max(
        max(pair_paths_count.values(), default=int(selected_pair.get("num_paths", 1))),
        int(selected_pair.get("max_num_paths", 1) or 1),
        int(selected_pair.get("num_paths", 1) or 1),
        1,
    )
)
print(f"\nDQN training across {len(pair_pool)} AS pair(s)")
print(f"  Action dim (max paths over pool): {action_dim}")


# -----------------------------------------------------------------------------
# Goodput cap from topology static bandwidth (P95 of per-path min-bandwidth).
# -----------------------------------------------------------------------------

_path_min_bws: List[float] = []
for pair in pair_pool:
    for p in path_store.find_paths(int(pair[0]), int(pair[1])) or []:
        sm = (p.get("static_metrics") if isinstance(p, dict) else None) or {}
        if "min_bandwidth" in sm:
            _path_min_bws.append(float(sm["min_bandwidth"]))
GOODPUT_CAP_MBPS = float(np.percentile(_path_min_bws, 95)) if _path_min_bws else 100.0
GOODPUT_CAP_MBPS = max(50.0, GOODPUT_CAP_MBPS)
print(f"  Goodput cap for reward normalization: {GOODPUT_CAP_MBPS:.0f} Mbps")


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
    per_hop_probe_cost_ms=0.5,
    per_hop_full_probe_cost_ms=20.0,
    pair_pool=pair_pool,
    episode_length=EPISODE_LENGTH,
    rng_seed=42,
)

state_dim = 5
config = EnhancedDQNConfig(
    learning_rate=1e-3,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10_000,
    min_buffer_size=500,
    batch_size=32,
    target_update_every=100,
    hidden_dim=128,
    n_hidden_layers=2,
    use_batch_norm=False,
    use_prioritized_replay=False,
    use_dueling_dqn=True,
    use_double_dqn=True,
    use_action_masking=True,
    tau=0.05,
)
agent = EnhancedDQNAgent(state_dim, action_dim, config)

# Reward weights from the simple_dqn paper (composite goodput + link trust).
W1, W2, W3, W4 = 0.7, 0.3, 0.5, 0.5

print("\nDQN configuration:")
print(f"  State dim: {state_dim}")
print(f"  Action dim: {action_dim}")
print(f"  Reward weights: w1={W1}, w2={W2}, w3={W3}, w4={W4}")
print(f"  Episode length (hours): {EPISODE_LENGTH}")
print(f"  Device: {agent.device}")


# -----------------------------------------------------------------------------
# State featurization
# -----------------------------------------------------------------------------


def _aggregate_state(env: EvaluationPathSelectionEnv, hour_idx: int) -> np.ndarray:
    """Build the 5-dim state vector from the env's current link states.

    Features:
        0. day_of_week (0..6) / 6
        1. hour_of_day (0..23) / 23
        2. mean utilization across the pair's paths
        3. mean link-trust score across the pair's paths
        4. fraction of paths with utilization > 0.7 (congestion signal)
    """
    # Mirror the timestamp scheme in 03_simulate_traffic.py: hour_idx counts
    # hours from the start of the simulation window.
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
# Training loop (multi-pair, stateful episodes)
# -----------------------------------------------------------------------------

# Use the first 14 days of the simulation window for training. We aim for a
# fixed number of total training steps that scales with the pair pool so the
# DQN sees each pair several times — a single pair's flow length is no longer
# a meaningful upper bound now that training is multi-pair.
TRAINING_HOURS = list(range(14 * 24))
TARGET_TRAINING_STEPS = max(2_000, min(50_000, 200 * len(pair_pool)))
NUM_EPISODES = max(50, TARGET_TRAINING_STEPS // EPISODE_LENGTH)

# Allow override via env var for very fast smoke tests / very long runs.
_env_eps = os.environ.get("DQN_TRAIN_EPISODES", "").strip()
if _env_eps.isdigit():
    NUM_EPISODES = int(_env_eps)

print(
    f"\nTraining: {NUM_EPISODES} episodes x {EPISODE_LENGTH} hours each "
    f"(~{NUM_EPISODES * EPISODE_LENGTH} steps)..."
)

random.seed(42)
training_pair_rng = random.Random(123)
training_hour_rng = random.Random(456)

episode_rewards: List[float] = []
episode_probes: List[int] = []
losses: List[float] = []
total_steps = 0

for ep in tqdm(range(NUM_EPISODES), desc="Episodes"):
    pair = training_pair_rng.choice(pair_pool)
    start_hour = training_hour_rng.choice(TRAINING_HOURS)

    env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=start_hour)
    state = _aggregate_state(env, env.hour_idx)
    mask = env.action_mask(action_dim)

    ep_reward = 0.0
    for step in range(EPISODE_LENGTH):
        action = int(agent.act(state, action_mask=mask))
        # Constrain to valid actions (defensive: act() should already mask).
        if not mask[action]:
            valid = np.where(mask)[0]
            action = int(valid[0]) if len(valid) > 0 else 0

        next_obs, _, done, info = env.step(action)
        next_state = _aggregate_state(env, env.hour_idx)
        next_mask = env.action_mask(action_dim)

        reward = _reward_from_metrics(info["path_metrics"], GOODPUT_CAP_MBPS)
        ep_reward += reward

        agent.remember(
            state, action, reward, next_state, done, mask, next_mask
        )

        if len(agent.memory) >= config.min_buffer_size:
            loss = agent.replay()
            if loss is not None:
                losses.append(float(loss))

        state = next_state
        mask = next_mask
        total_steps += 1
        if done:
            break

    agent.epsilon = max(config.epsilon_end, agent.epsilon * config.epsilon_decay)
    agent.episodes += 1
    episode_rewards.append(ep_reward / max(1, EPISODE_LENGTH))
    episode_probes.append(int(env.num_latency_probes + env.num_bandwidth_probes))


# -----------------------------------------------------------------------------
# Save model + stats
# -----------------------------------------------------------------------------

model_file = run_path / "dqn_model.pth"
torch.save(
    {
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict(),
        "epsilon": agent.epsilon,
        "steps": agent.steps,
        "episodes": agent.episodes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "config": config,
        "goodput_cap_mbps": GOODPUT_CAP_MBPS,
        "pair_pool": [list(p) for p in pair_pool],
        "reward_weights": {"w1": W1, "w2": W2, "w3": W3, "w4": W4},
    },
    model_file,
)
print(f"\nModel saved to: {model_file}")

training_stats = {
    "num_episodes": NUM_EPISODES,
    "episode_length_hours": EPISODE_LENGTH,
    "total_steps": total_steps,
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
    "pair_pool_size": len(pair_pool),
    "action_dim": action_dim,
}
with open(run_path / "training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=2)

print("\nTraining statistics:")
print(f"  Average reward (per step):      {training_stats['avg_reward']:.3f}")
print(f"  Average probes per episode:     {training_stats['avg_probes_per_episode']:.1f}")
print(f"  Final epsilon:                  {training_stats['final_epsilon']:.3f}")
print("\nTraining complete!")
