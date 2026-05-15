#!/usr/bin/env python3
"""Train the **enhanced** path-scoring DQN (Dueling + PER + Double DQN).

Dict observations ``{"global": (G,), "paths": (N, P)}`` with variable ``N``.
For the flat MLP baseline without PER/Dueling, see ``04_train_scoring_dqn.py``.
"""

from __future__ import annotations

import json
import os as _os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from _common import resolve_run_dir, topology_dir
from tqdm import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from scoring_dqn_state import GLOBAL_DIM, PATH_FEATURE_DIM, dict_state

from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
from src.simulation.evaluation_env import EvaluationPathSelectionEnv
from src.simulation.path_store import InMemoryPathStore

run_dir = resolve_run_dir()
run_path = Path(run_dir)

_topo_json = topology_dir(run_path) / "scion_topology.json"
if not _topo_json.is_file():
    _leg = run_path / "scion_topology.json"
    _topo_json = _leg if _leg.is_file() else _topo_json
with open(_topo_json, "r") as f:
    topology_data = json.load(f)

with open(run_path / "selected_pair.json", "r") as f:
    selected_pair = json.load(f)

path_store = InMemoryPathStore.load(run_path / "path_store.json")

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

_min_bws: List[float] = []
for pair in pair_pool:
    for p in path_store.find_paths(int(pair[0]), int(pair[1])) or []:
        sm = p.get("static_metrics") or {}
        if "min_bandwidth" in sm:
            _min_bws.append(float(sm["min_bandwidth"]))
GOODPUT_CAP_MBPS = max(50.0, float(np.percentile(_min_bws, 95)) if _min_bws else 100.0)

print(f"\nTraining Enhanced Path-Scoring DQN across {len(pair_pool)} AS pair(s)")
print(f"  Global dim={GLOBAL_DIM}, path feature dim={PATH_FEATURE_DIM}")
print(f"  Goodput cap: {GOODPUT_CAP_MBPS:.0f} Mbps")

EPISODE_LENGTH = 24
env = EvaluationPathSelectionEnv(
    topology_data=topology_data,
    path_store=path_store,
    link_states=link_states,
    latency_probe_cost_ms=10.0,
    bandwidth_probe_cost_ms=100.0,
    pair_pool=pair_pool,
    episode_length=EPISODE_LENGTH,
    rng_seed=42,
)

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
    use_prioritized_replay=True,
    use_dueling_dqn=True,
    use_double_dqn=True,
    tau=0.05,
)
agent = EnhancedPathScoringDQNAgent(
    global_dim=GLOBAL_DIM,
    path_dim=PATH_FEATURE_DIM,
    config=config,
)

W1, W2, W3, W4 = 0.7, 0.3, 0.5, 0.5
print("\nEnhanced Path-Scoring DQN configuration:")
print(f"  Dueling: {config.use_dueling_dqn}, PER: {config.use_prioritized_replay}")
print(f"  Double DQN: {config.use_double_dqn}")
print(f"  Episode length (hours): {EPISODE_LENGTH}")
print(f"  Device: {agent.device}")


def _reward_from_metrics(path_metrics: Dict, env: EvaluationPathSelectionEnv) -> float:
    bw = float(path_metrics.get("bandwidth_mbps") or 0.0)
    states = list(env.current_link_states.values())
    if states:
        max_possible_bw = max(float(s.get("available_bandwidth_mbps", 1.0)) for s in states)
        max_possible_bw = max(max_possible_bw, 1.0)
    else:
        max_possible_bw = 100.0
    goodput = max(0.0, min(bw / max_possible_bw, 1.0))
    loss = float(path_metrics.get("loss_rate", 0.0))
    delay = min(100.0, float(path_metrics.get("latency_ms", 50.0))) / 100.0
    trust = max(0.0, min(1.0, 1.0 - (W3 * loss + W4 * delay)))
    return float(2.0 * (W1 * goodput + W2 * trust) - 1.0)


TRAINING_HOURS = list(range(14 * 24))
TARGET_TRAINING_STEPS = max(2_000, min(50_000, 200 * len(pair_pool)))
NUM_EPISODES = max(50, TARGET_TRAINING_STEPS // EPISODE_LENGTH)

_env_eps = _os.environ.get("DQN_TRAIN_EPISODES", "").strip()
if _env_eps.isdigit():
    NUM_EPISODES = int(_env_eps)

print(
    f"\nTraining: {NUM_EPISODES} episodes x {EPISODE_LENGTH} hours each "
    f"(~{NUM_EPISODES * EPISODE_LENGTH} steps)..."
)

training_pair_rng = random.Random(123)
training_hour_rng = random.Random(456)
episode_rewards: List[float] = []
episode_probes: List[int] = []
losses: List[float] = []

for ep in tqdm(range(NUM_EPISODES), desc="Episodes"):
    pair = training_pair_rng.choice(pair_pool)
    start_hour = training_hour_rng.choice(TRAINING_HOURS)
    env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=start_hour)
    state = dict_state(env, env.hour_idx, W3, W4)
    ep_reward = 0.0
    for _step in range(EPISODE_LENGTH):
        if state["paths"].shape[0] == 0:
            break
        action = int(agent.act(state))
        if action >= len(env.available_paths):
            action = 0
        _, _, done, info = env.step(action)
        next_state = dict_state(env, env.hour_idx, W3, W4)
        reward = _reward_from_metrics(info["path_metrics"], env)
        ep_reward += reward
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        if loss is not None:
            losses.append(float(loss))
        state = next_state
        if done:
            break
    agent.update_epsilon()
    episode_rewards.append(ep_reward / max(1, EPISODE_LENGTH))
    episode_probes.append(int(env.num_latency_probes + env.num_bandwidth_probes))

model_file = run_path / "dqn_scoring_enhanced_model.pth"
torch.save(
    {
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict(),
        "epsilon": agent.epsilon,
        "steps": agent.steps,
        "episodes": agent.episodes,
        "beta": agent.beta,
        "config": config,
        "global_dim": GLOBAL_DIM,
        "path_dim": PATH_FEATURE_DIM,
        "goodput_cap_mbps": GOODPUT_CAP_MBPS,
        "reward_weights": {"w1": W1, "w2": W2, "w3": W3, "w4": W4},
        "pair_pool": [list(p) for p in pair_pool],
    },
    model_file,
)
print(f"\nEnhanced path-scoring model saved to: {model_file}")

training_stats = {
    "num_episodes": NUM_EPISODES,
    "episode_length_hours": EPISODE_LENGTH,
    "episode_rewards": episode_rewards,
    "episode_probes": episode_probes,
    "losses": losses,
    "final_epsilon": agent.epsilon,
    "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    "avg_probes_per_episode": float(np.mean(episode_probes)) if episode_probes else 0.0,
    "reward_weights": {"w1": W1, "w2": W2, "w3": W3, "w4": W4},
    "goodput_cap_mbps": GOODPUT_CAP_MBPS,
    "pair_pool_size": len(pair_pool),
    "global_dim": GLOBAL_DIM,
    "path_dim": PATH_FEATURE_DIM,
}
with open(run_path / "dqn_scoring_enhanced_training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=4)
print(
    f"Enhanced path-scoring training stats saved to: "
    f"{run_path / 'dqn_scoring_enhanced_training_stats.json'}"
)
