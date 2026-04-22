#!/usr/bin/env python3
"""
Evaluate all path selection methods on the last 14 days
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.simulation.evaluation_env import EvaluationPathSelectionEnv
from src.baselines.shortest_path import ShortestPathSelector
from src.baselines.widest_path import WidestPathSelector
from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.ecmp import ECMPSelector
from src.baselines.random_selection import RandomSelector
from src.baselines.scion_default import SCIONDefaultSelector

# Get run directory
if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    dirs = [d for d in os.listdir('.') if d.startswith('run_')]
    run_dir = sorted(dirs)[-1]

print(f"Using run directory: {run_dir}")

# Load all necessary data
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

# Load trained DQN model
_model_path = os.path.join(run_dir, "dqn_model.pth")
try:
    model_checkpoint = torch.load(_model_path, map_location="cpu", weights_only=False)
except TypeError:
    model_checkpoint = torch.load(_model_path, map_location="cpu")

src_as = selected_pair['source_as']
dst_as = selected_pair['destination_as']
num_paths = selected_pair['num_paths']

# Get evaluation flows (last 14 days)
eval_flows = [f for f in traffic_flows if f['day'] >= 14]
print(f"\nEvaluation samples: {len(eval_flows)}")

# Create environment
env = EvaluationPathSelectionEnv(
    topology_data=topology_data,
    path_store=path_store,
    link_states=link_states,
    latency_probe_cost_ms=10.0,
    bandwidth_probe_cost_ms=100.0,
)

# Initialize DQN agent (checkpoint from ``04_train_dqn.py``)
_config = model_checkpoint["config"]
_state_dim = int(model_checkpoint.get("state_dim", 5))
_action_dim = int(model_checkpoint.get("action_dim", num_paths))
dqn_agent = EnhancedDQNAgent(_state_dim, _action_dim, _config)
_q = model_checkpoint.get("q_network") or model_checkpoint.get("model_state_dict")
dqn_agent.q_network.load_state_dict(_q)
if "target_network" in model_checkpoint:
    dqn_agent.target_network.load_state_dict(model_checkpoint["target_network"])
if "optimizer" in model_checkpoint:
    try:
        dqn_agent.optimizer.load_state_dict(model_checkpoint["optimizer"])
    except Exception:
        pass
if "scheduler" in model_checkpoint:
    try:
        dqn_agent.scheduler.load_state_dict(model_checkpoint["scheduler"])
    except Exception:
        pass
dqn_agent.epsilon = 0.0  # No exploration during evaluation

# Initialize baseline methods
baseline_methods = {
    'shortest_path': ShortestPathSelector(),
    'widest_path': WidestPathSelector(),
    'lowest_latency': LowestLatencySelector(),
    'ecmp': ECMPSelector(),
    'random': RandomSelector(),
    'scion_default': SCIONDefaultSelector()
}

# Evaluation metrics storage
results = defaultdict(lambda: {
    'rewards': [],
    'latencies': [],
    'bandwidths': [],
    'losses': [],
    'latency_probes': 0,
    'bandwidth_probes': 0,
    'total_probe_time_ms': 0,
    'selection_times_ms': []
})

# Reward weights from training
w1, w2, w3, w4 = 0.7, 0.3, 0.5, 0.5

probe_reduction = 0.0
time_reduction = 0.0

print("\nEvaluating methods...")

# Evaluate each method
for method_name, method in list(baseline_methods.items()) + [('dqn', dqn_agent)]:
    print(f"\nEvaluating {method_name}...")
    
    method_results = results[method_name]
    
    for i, flow in enumerate(tqdm(eval_flows, desc=f"{method_name}")):
        # Reset environment
        state = env.reset(source_as=src_as, dest_as=dst_as)
        env.current_link_states = link_states[flow['hour'] + flow['day'] * 24]
        
        # Get available paths
        paths = env.available_paths
        
        # Time the selection
        start_time = time.time()
        path_metrics = {}
        
        if method_name == 'dqn':
            # DQN with selective probing
            # Extract state features
            state_features = [
                flow['day_of_week'] / 6.0,
                flow['hour'] / 23.0,
                flow['bandwidth_mbps'] / 1000.0,
                0.5,  # Average utilization (placeholder)
                0.7   # Average trust (placeholder)
            ]
            state = np.array(state_features, dtype=np.float32)
            
            # Select action
            action = dqn_agent.act(state)
            
            # DQN only probes the selected path
            if action < len(paths):
                path_metrics = env.probe_path_full(action)
                method_results['bandwidth_probes'] += 1
                _sm = getattr(paths[action], "static_metrics", {}) or {}
                method_results['total_probe_time_ms'] += 100 + 20 * int(_sm.get("hop_count", 1))
            
        else:
            # Baseline methods must probe ALL paths
            path_metrics_list = []
            
            for path_idx in range(len(paths)):
                # Always probe latency
                latency_metrics = env.probe_path_latency(path_idx)
                method_results['latency_probes'] += 1
                _sm = getattr(paths[path_idx], "static_metrics", {}) or {}
                method_results['total_probe_time_ms'] += 10 + 0.5 * int(_sm.get("hop_count", 1))
                
                # Methods that need bandwidth also probe it
                if method_name in ['widest_path', 'ecmp']:
                    bw_metrics = env.probe_path_full(path_idx)
                    method_results['bandwidth_probes'] += 1
                    _sm2 = getattr(paths[path_idx], "static_metrics", {}) or {}
                    method_results['total_probe_time_ms'] += 100 + 20 * int(_sm2.get("hop_count", 1))
                    path_metrics_list.append(bw_metrics)
                else:
                    path_metrics_list.append(latency_metrics)
            
            flow_stub = {'src': src_as, 'dst': dst_as}
            state_stub = np.zeros(1, dtype=np.float32)
            # Select path based on method
            if method_name == 'shortest_path':
                action = method.select_path(paths, path_metrics_list, flow_stub, state_stub)
            elif method_name == 'widest_path':
                action = method.select_path(paths, path_metrics_list, flow_stub, state_stub)
            elif method_name == 'lowest_latency':
                action = method.select_path(paths, path_metrics_list, flow_stub, state_stub)
            elif method_name == 'ecmp':
                action = method.select_path(paths, path_metrics_list, flow_stub, state_stub)
            elif method_name == 'random':
                action = np.random.choice(len(paths))
            else:  # scion_default
                action = method.select_path(paths, path_metrics_list, flow_stub, state_stub)
            
            if action < len(paths):
                path_metrics = path_metrics_list[action]
        
        selection_time_ms = (time.time() - start_time) * 1000
        method_results['selection_times_ms'].append(selection_time_ms)
        
        # Calculate reward
        if action < len(paths) and 'path_metrics' in locals():
            # Get metrics for selected path
            latency = path_metrics.get('latency_ms', 50)
            bandwidth = path_metrics.get('bandwidth_mbps')
            if bandwidth is None:
                bandwidth = 0.0
            loss_rate = path_metrics.get('loss_rate', 0.0)
            
            # Calculate reward as per simple_dqn.tex
            goodput = min(bandwidth, 50) / 50.0
            delay_normalized = min(latency, 100) / 100.0
            link_trust = 1 - (w3 * loss_rate + w4 * delay_normalized)
            reward = 2 * (w1 * goodput + w2 * link_trust) - 1
            
            # Store results
            method_results['rewards'].append(reward)
            method_results['latencies'].append(latency)
            method_results['bandwidths'].append(bandwidth)
            method_results['losses'].append(loss_rate)

# Calculate statistics
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

summary = {}

for method_name, method_results in results.items():
    if not method_results['rewards']:
        continue
        
    rewards = np.array(method_results['rewards'])
    latencies = np.array(method_results['latencies'])
    bandwidths = np.array(method_results['bandwidths'])
    probe_time = method_results['total_probe_time_ms']
    selection_times = np.array(method_results['selection_times_ms'])
    
    summary[method_name] = {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'reward_p50': float(np.percentile(rewards, 50)),
        'reward_p95': float(np.percentile(rewards, 95)),
        'latency_mean': float(np.mean(latencies)),
        'latency_p50': float(np.percentile(latencies, 50)),
        'latency_p95': float(np.percentile(latencies, 95)),
        'bandwidth_mean': float(np.mean(bandwidths)),
        'latency_probes': method_results['latency_probes'],
        'bandwidth_probes': method_results['bandwidth_probes'],
        'total_probes': method_results['latency_probes'] + method_results['bandwidth_probes'],
        'total_probe_time_ms': probe_time,
        'avg_probe_time_per_selection': probe_time / len(eval_flows),
        'avg_selection_time_ms': float(np.mean(selection_times))
    }
    
    print(f"\n{method_name.upper()}:")
    print(f"  Reward: {summary[method_name]['reward_mean']:.3f} ± {summary[method_name]['reward_std']:.3f}")
    print(f"  Latency (ms): {summary[method_name]['latency_mean']:.1f} (p95: {summary[method_name]['latency_p95']:.1f})")
    print(f"  Bandwidth (Mbps): {summary[method_name]['bandwidth_mean']:.1f}")
    print(f"  Total probes: {summary[method_name]['total_probes']} ({summary[method_name]['latency_probes']} latency, {summary[method_name]['bandwidth_probes']} bandwidth)")
    print(f"  Avg probe overhead per selection: {summary[method_name]['avg_probe_time_per_selection']:.1f} ms")
    print(f"  Avg selection time: {summary[method_name]['avg_selection_time_ms']:.1f} ms")

# Calculate probe reduction for DQN
if 'dqn' in summary and len(eval_flows) > 0:
    baseline_avg_probes = np.mean([s['total_probes'] for k, s in summary.items() if k != 'dqn'])
    dqn_probes = summary['dqn']['total_probes']
    if baseline_avg_probes:
        probe_reduction = (baseline_avg_probes - dqn_probes) / baseline_avg_probes * 100
    
    baseline_avg_time = np.mean([s['total_probe_time_ms'] for k, s in summary.items() if k != 'dqn'])
    dqn_time = summary['dqn']['total_probe_time_ms']
    if baseline_avg_time:
        time_reduction = (baseline_avg_time - dqn_time) / baseline_avg_time * 100
    
    print(f"\n{'='*60}")
    print("DQN PROBE REDUCTION:")
    print(f"  Probe count reduction: {probe_reduction:.1f}%")
    print(f"  Probe time reduction: {time_reduction:.1f}%")
    print(f"  DQN probes per selection: {dqn_probes/len(eval_flows):.1f}")
    print(f"  Baseline avg probes per selection: {baseline_avg_probes/len(eval_flows):.1f}")

# Save results
results_file = os.path.join(run_dir, "evaluation_results.json")
with open(results_file, 'w') as f:
    json.dump({
        'summary': summary,
        'num_eval_flows': len(eval_flows),
        'num_paths': num_paths,
        'probe_reduction_percent': probe_reduction if 'dqn' in summary else 0,
        'time_reduction_percent': time_reduction if 'dqn' in summary else 0
    }, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("\nEvaluation complete!")