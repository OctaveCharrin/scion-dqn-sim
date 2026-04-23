#!/usr/bin/env python3
"""
Simulate 28 days of traffic on the SCION network
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

from _common import resolve_run_dir, topology_dir

run_dir = resolve_run_dir()

# Load topology and selected pair
_topo_json = topology_dir(run_dir) / "scion_topology.json"
if not _topo_json.is_file():
    _legacy = os.path.join(run_dir, "scion_topology.json")
    _topo_json = Path(_legacy) if os.path.isfile(_legacy) else _topo_json
with open(_topo_json, "r") as f:
    topology_data = json.load(f)

with open(os.path.join(run_dir, "selected_pair.json"), 'r') as f:
    selected_pair = json.load(f)

with open(os.path.join(run_dir, "path_store.pkl"), 'rb') as f:
    path_store = pickle.load(f)

src_as = selected_pair['source_as']
dst_as = selected_pair['destination_as']

print(f"\nSimulating traffic for AS {src_as} -> AS {dst_as}")
print(f"Number of available paths: {selected_pair['num_paths']}")

# Generate traffic for 28 days
print("\nGenerating 28 days of traffic...")
num_days = 28
samples_per_day = 24  # Hourly samples
total_samples = num_days * samples_per_day

# Create traffic flows with diurnal pattern
np.random.seed(42)
flows = []
timestamps = []

start_time = datetime.now()
current_time = start_time

for day in range(num_days):
    for hour in range(24):
        timestamp = current_time + timedelta(days=day, hours=hour)
        timestamps.append(timestamp)
        
        # Diurnal pattern: peak at 14:00, trough at 02:00
        hour_of_day = hour
        base_rate = 100  # Mbps
        
        # Sinusoidal pattern
        diurnal_factor = 1 + 0.5 * np.sin((hour_of_day - 2) * np.pi / 12)
        
        # Weekly pattern: lower on weekends
        day_of_week = timestamp.weekday()
        weekly_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Random variation
        random_factor = np.random.uniform(0.8, 1.2)
        
        # Calculate flow rate
        flow_rate = base_rate * diurnal_factor * weekly_factor * random_factor
        
        # Create flow
        flow = {
            'timestamp': timestamp,
            'source_as': src_as,
            'destination_as': dst_as,
            'bandwidth_mbps': flow_rate,
            'duration_s': 3600,  # 1 hour
            'day': day,
            'hour': hour,
            'day_of_week': day_of_week
        }
        flows.append(flow)

print(f"Generated {len(flows)} traffic flows")

# Save traffic flows
traffic_file = os.path.join(run_dir, "traffic_flows.pkl")
with open(traffic_file, 'wb') as f:
    pickle.dump(flows, f)

# Also save as CSV for analysis
df = pd.DataFrame(flows)
df.to_csv(os.path.join(run_dir, "traffic_flows.csv"), index=False)

print(f"Traffic flows saved to: {traffic_file}")

# Simulate link states based on traffic
print("\nSimulating link states...")

# Get all paths between source and destination
paths = path_store.find_paths(src_as, dst_as)
print(f"Found {len(paths)} paths to simulate")

# Initialize link states
link_states = {}
link_history = []

for i, flow in enumerate(tqdm(flows, desc="Simulating link states")):
    timestamp = flow['timestamp']
    
    # Update link states for all links in all paths
    hourly_states = {}
    
    for path_idx, path in enumerate(paths):
        path_key = f"path_{path_idx}"
        
        # Calculate path metrics based on time of day and load
        base_latency = sum(hop.get('latency', 10) for hop in path['hops'])
        base_bandwidth = min(hop.get('bandwidth', 1000) for hop in path['hops'])
        
        # Add congestion based on time of day
        congestion_factor = 1 + 0.3 * flow['bandwidth_mbps'] / base_rate
        
        # Latency increases with congestion
        current_latency = base_latency * congestion_factor
        
        # Available bandwidth decreases with load
        utilization = min(0.9, flow['bandwidth_mbps'] / base_bandwidth)
        available_bandwidth = base_bandwidth * (1 - utilization)
        
        # Packet loss increases with high utilization
        if utilization > 0.8:
            loss_rate = 0.001 * (utilization - 0.8) / 0.2
        else:
            loss_rate = 0.0
        
        hourly_states[path_key] = {
            'latency_ms': current_latency,
            'available_bandwidth_mbps': available_bandwidth,
            'utilization': utilization,
            'loss_rate': loss_rate,
            'hop_count': len(path['hops'])
        }
    
    link_states[i] = hourly_states
    
    # Record history
    link_history.append({
        'timestamp': timestamp,
        'hour_idx': i,
        'states': hourly_states
    })

# Save link states
link_states_file = os.path.join(run_dir, "link_states.pkl")
with open(link_states_file, 'wb') as f:
    pickle.dump(link_states, f)

print(f"Link states saved to: {link_states_file}")

# Create summary statistics
print("\nTraffic simulation summary:")
print(f"  - Total days: {num_days}")
print(f"  - Total flows: {len(flows)}")
print(f"  - Average bandwidth: {df['bandwidth_mbps'].mean():.2f} Mbps")
print(f"  - Peak bandwidth: {df['bandwidth_mbps'].max():.2f} Mbps")
print(f"  - Training period: Days 1-14 ({14 * samples_per_day} samples)")
print(f"  - Evaluation period: Days 15-28 ({14 * samples_per_day} samples)")

# Save simulation metadata
metadata = {
    'source_as': src_as,
    'destination_as': dst_as,
    'num_paths': len(paths),
    'num_days': num_days,
    'samples_per_day': samples_per_day,
    'total_samples': total_samples,
    'training_samples': 14 * samples_per_day,
    'evaluation_samples': 14 * samples_per_day,
    'traffic_stats': {
        'mean_bandwidth_mbps': float(df['bandwidth_mbps'].mean()),
        'std_bandwidth_mbps': float(df['bandwidth_mbps'].std()),
        'min_bandwidth_mbps': float(df['bandwidth_mbps'].min()),
        'max_bandwidth_mbps': float(df['bandwidth_mbps'].max())
    }
}

metadata_file = os.path.join(run_dir, "simulation_metadata.json")
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSimulation metadata saved to: {metadata_file}")
print("\nTraffic simulation complete!")