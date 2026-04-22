#!/usr/bin/env python3
"""
Train DQN agent following the approach in simple_dqn.tex
"""

import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm

from _common import resolve_run_dir

from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.simulation.evaluation_env import EvaluationPathSelectionEnv

run_dir = resolve_run_dir()

# Load data
with open(os.path.join(run_dir, "scion_topology.json"), 'r') as f:
    topology_data = json.load(f)

with open(os.path.join(run_dir, "selected_pair.json"), 'r') as f:
    selected_pair = json.load(f)

with open(os.path.join(run_dir, "path_store.pkl"), 'rb') as f:
    path_store = pickle.load(f)

with open(os.path.join(run_dir, "traffic_flows.pkl"), 'rb') as f:
    traffic_flows = pickle.load(f)

with open(os.path.join(run_dir, "link_states.pkl"), 'rb') as f:
    link_states = pickle.load(f)

src_as = selected_pair['source_as']
dst_as = selected_pair['destination_as']
num_paths = selected_pair['num_paths']

print(f"\nTraining DQN for AS {src_as} -> AS {dst_as}")
print(f"Number of paths: {num_paths}")

# Get training flows (first 14 days)
training_flows = [f for f in traffic_flows if f['day'] < 14]
print(f"Training samples: {len(training_flows)}")

# Create environment with selective probing (evaluation pipeline adapter)
env = EvaluationPathSelectionEnv(
    topology_data=topology_data,
    path_store=path_store,
    link_states=link_states,
    latency_probe_cost_ms=10.0,
    bandwidth_probe_cost_ms=100.0,
    probe_type="adaptive",
)

# Configure DQN (matches ``EnhancedDQNAgent`` / ``EnhancedDQNConfig``)
state_dim = 5
epsilon_decay = 0.995
config = EnhancedDQNConfig(
    learning_rate=1e-3,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    buffer_size=10000,
    min_buffer_size=500,
    batch_size=32,
    target_update_every=100,
    hidden_dim=128,
    n_hidden_layers=2,
    use_batch_norm=False,
    use_prioritized_replay=False,
    tau=0.05,
)

agent = EnhancedDQNAgent(state_dim, num_paths, config)

# Training parameters from simple_dqn.tex
w1 = 0.7  # Weight for goodput
w2 = 0.3  # Weight for link trust
w3 = 0.5  # Weight for packet loss in trust calculation
w4 = 0.5  # Weight for delay in trust calculation

print(f"\nDQN Configuration:")
print(f"  State dimensions: {state_dim}")
print(f"  Action space: {num_paths} paths")
print(f"  Reward weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}")
print(f"  Device: {agent.device}")

# Training loop
print("\nTraining DQN agent...")
episode_rewards = []
episode_probes = []
losses = []

for episode in tqdm(range(len(training_flows)), desc="Training episodes"):
    flow = training_flows[episode]
    
    # Reset environment with this flow's link states
    state = env.reset(source_as=src_as, dest_as=dst_as)
    env.current_link_states = link_states[flow['hour'] + flow['day'] * 24]
    
    # Extract enhanced state features as per simple_dqn.tex
    state_features = []
    state_features.append(flow['day_of_week'] / 6.0)  # Day of week normalized
    state_features.append(flow['hour'] / 23.0)  # Time of day normalized
    state_features.append(flow['bandwidth_mbps'] / 1000.0)  # Link bandwidth normalized
    
    # Get average utilization and trust from available paths
    avg_utilization = 0
    avg_trust = 0
    for i, path_states in enumerate(env.current_link_states.values()):
        avg_utilization += path_states.get('utilization', 0.5)
        # Calculate link trust as per simple_dqn.tex
        packet_loss = path_states.get('loss_rate', 0.0)
        delay_normalized = path_states.get('latency_ms', 50) / 100.0
        trust = 1 - (w3 * packet_loss + w4 * delay_normalized)
        avg_trust += trust
    
    avg_utilization /= len(env.current_link_states)
    avg_trust /= len(env.current_link_states)
    
    state_features.append(avg_utilization)  # Link capacity utilization
    state_features.append(avg_trust)  # Link trust score
    
    state = np.array(state_features, dtype=np.float32)
    
    # Select action using epsilon-greedy
    action = agent.act(state)
    
    # Take action in environment
    next_state, reward, done, info = env.step(action)
    
    # Calculate reward as per simple_dqn.tex
    path_metrics = info.get('path_metrics', {})
    
    # Goodput normalized to [0, 1] based on max 50 Mbps
    goodput = min(path_metrics.get('bandwidth_mbps', 0), 50) / 50.0
    
    # Link trust calculation
    packet_loss = path_metrics.get('loss_rate', 0.0)
    delay_normalized = min(path_metrics.get('latency_ms', 50), 100) / 100.0
    link_trust = 1 - (w3 * packet_loss + w4 * delay_normalized)
    
    # Final reward: r = 2(w1 * G + w2 * T) - 1
    reward = 2 * (w1 * goodput + w2 * link_trust) - 1
    
    # Store transition
    agent.remember(state, action, reward, next_state, done)
    
    # Train agent
    if len(agent.memory) >= config.min_buffer_size:
        loss = agent.replay()
        if loss is not None:
            losses.append(loss)
    
    # Record metrics
    episode_rewards.append(reward)
    episode_probes.append(info.get('probe_count', 0))
    
    # Update epsilon
    agent.epsilon = max(config.epsilon_end, agent.epsilon * epsilon_decay)

# Save trained model
model_file = os.path.join(run_dir, "dqn_model.pth")
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
        "action_dim": num_paths,
        "config": config,
    },
    model_file,
)

print(f"\nModel saved to: {model_file}")

# Save training statistics
training_stats = {
    'num_episodes': len(training_flows),
    'episode_rewards': episode_rewards,
    'episode_probes': episode_probes,
    'losses': losses,
    'final_epsilon': agent.epsilon,
    'avg_reward': np.mean(episode_rewards),
    'avg_probes_per_episode': np.mean(episode_probes),
    'reward_weights': {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}
}

stats_file = os.path.join(run_dir, "training_stats.json")
with open(stats_file, 'w') as f:
    json.dump(training_stats, f, indent=2)

print(f"\nTraining statistics:")
print(f"  Average reward: {training_stats['avg_reward']:.3f}")
print(f"  Average probes per selection: {training_stats['avg_probes_per_episode']:.1f}")
print(f"  Final epsilon: {training_stats['final_epsilon']:.3f}")
print(f"\nTraining complete!")