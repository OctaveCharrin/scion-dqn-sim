#!/usr/bin/env python3
"""
Run the complete evaluation pipeline
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, run_dir=None):
    """Run a script and check for errors"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    cmd = [sys.executable, script_name]
    if run_dir:
        cmd.append(run_dir)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR in {script_name}:")
        print(result.stderr)
        sys.exit(1)
    
    print(result.stdout)
    print(f"Completed in {elapsed:.1f} seconds")
    return result.stdout

run_dir = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(run_dir, exist_ok=True)
print(f"Using run directory: {run_dir}")

# Step 1: Generate topology
output = run_script('01_setup_dense_topology.py', run_dir)

# Step 2: Run beaconing
run_script('02_run_beaconing.py', run_dir)

# Step 3: Simulate traffic
run_script('03_simulate_traffic.py', run_dir)

# Step 4: Train DQN
run_script('04_train_dqn.py', run_dir)

# Step 5: Evaluate methods
run_script('05_evaluate_methods.py', run_dir)

# Step 6: Generate figures
run_script('05_generate_figures_final.py', run_dir)

print(f"\n{'='*60}")
print("EVALUATION COMPLETE!")
print(f"{'='*60}")
print(f"\nAll results saved in: {run_dir}/")
print("\nKey outputs:")
print(f"  - {run_dir}/scion_topology.json: Network topology")
print(f"  - {run_dir}/selected_pair.json: Source-destination pair")
print(f"  - {run_dir}/dqn_model.pth: Trained DQN model")
print(f"  - {run_dir}/evaluation_results.json: Performance comparison")
print(f"  - {run_dir}/figure1_probe_overhead.pdf: Probe overhead comparison")
print(f"  - {run_dir}/figure2_path_reward.pdf: Path reward distribution")
print(f"  - {run_dir}/figure3_probe_breakdown.pdf: Probe type breakdown")