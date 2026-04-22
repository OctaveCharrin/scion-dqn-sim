"""
Enhanced SCION environment with selective probing and differentiated costs
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from .environment_realistic import RealisticSCIONPathSelectionEnv

logger = logging.getLogger(__name__)


class SelectiveProbingSCIONEnv(RealisticSCIONPathSelectionEnv):
    """
    Enhanced environment with:
    - Differentiated costs for latency vs bandwidth probing
    - Selective probing for RL agents
    - Proper tracking of probing overhead
    """
    
    def __init__(self, *args, 
                 latency_probe_cost_ms: float = 10.0,
                 bandwidth_probe_cost_ms: float = 100.0,
                 bandwidth_probe_bw_cost_mbps: float = 10.0,
                 probe_type: str = 'full',  # 'full', 'latency_only', 'adaptive'
                 **kwargs):
        """
        Initialize selective probing environment
        
        Args:
            latency_probe_cost_ms: Time cost for latency probe (ping)
            bandwidth_probe_cost_ms: Time cost for bandwidth probe (iperf-style)
            bandwidth_probe_bw_cost_mbps: Bandwidth consumed by bandwidth probe
            probe_type: Type of probing to perform
        """
        # Override probe overhead with latency probe cost by default
        kwargs['probe_overhead_ms'] = latency_probe_cost_ms
        kwargs['probe_bandwidth_cost_mbps'] = 0.1  # Minimal for latency probe
        
        super().__init__(*args, **kwargs)
        
        self.latency_probe_cost_ms = latency_probe_cost_ms
        self.bandwidth_probe_cost_ms = bandwidth_probe_cost_ms
        self.bandwidth_probe_bw_cost_mbps = bandwidth_probe_bw_cost_mbps
        self.probe_type = probe_type
        
        # Track different types of probes
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.latency_probe_time_ms = 0.0
        self.bandwidth_probe_time_ms = 0.0
        
        # RL exploration tracking
        self.exploration_budget = 2  # Number of extra paths to probe for exploration
        self.path_probe_history = {}  # Track when paths were last probed
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment including probe tracking."""
        state, info = super().reset(seed=seed, options=options, **kwargs)

        # Reset probe counters
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.latency_probe_time_ms = 0.0
        self.bandwidth_probe_time_ms = 0.0

        return state, info
    
    def probe_path_latency(self, path_index: int) -> Dict:
        """
        Probe only latency for a path (cheap probe)
        """
        if path_index >= len(self.available_paths):
            return {
                'latency_ms': float('inf'),
                'bandwidth_mbps': None,  # Not measured
                'loss_rate': 0.01,  # Estimate
                'hop_count': 0,
                'probe_type': 'none'
            }
        
        # Check cache for recent latency probe
        if path_index in self.probed_path_metrics:
            cached = self.probed_path_metrics[path_index]
            if cached.get('probe_type') in ['latency', 'full']:
                return cached
        
        # Perform latency probe
        path = self.available_paths[path_index]
        adapted_path = self._adapt_path(path)
        
        # Get latency only (simulated cheap probe)
        metrics = self.pathprobe.probe(
            adapted_path,
            t_idx=self.current_time_slot,
            noisy=True
        )
        
        # Create latency-only result
        probed_metrics = {
            'latency_ms': metrics.latency_ms + self.latency_probe_cost_ms,
            'bandwidth_mbps': None,  # Not measured
            'loss_rate': metrics.loss_rate,  # Can be estimated from latency probe
            'hop_count': len(path.as_sequence),
            'probe_time': self.current_time_slot * 900,
            'probe_type': 'latency'
        }
        
        # Track latency probing cost
        self.num_latency_probes += 1
        self.latency_probe_time_ms += self.latency_probe_cost_ms
        self.total_probes += 1
        self.total_probe_time_ms += self.latency_probe_cost_ms
        
        # Cache result
        self.probed_path_metrics[path_index] = probed_metrics
        
        return probed_metrics
    
    def probe_path_full(self, path_index: int) -> Dict:
        """
        Probe both latency and bandwidth for a path (expensive probe)
        """
        if path_index >= len(self.available_paths):
            return {
                'latency_ms': float('inf'),
                'bandwidth_mbps': 0,
                'loss_rate': 1.0,
                'hop_count': 0,
                'probe_type': 'none'
            }
        
        # Check cache for recent full probe
        if path_index in self.probed_path_metrics:
            cached = self.probed_path_metrics[path_index]
            if cached.get('probe_type') == 'full':
                return cached
        
        # Perform full probe
        path = self.available_paths[path_index]
        adapted_path = self._adapt_path(path)
        
        # Get full metrics
        metrics = self.pathprobe.probe(
            adapted_path,
            t_idx=self.current_time_slot,
            noisy=True
        )
        
        # Create full result with both probe overheads
        probed_metrics = {
            'latency_ms': metrics.latency_ms + self.latency_probe_cost_ms,
            'bandwidth_mbps': max(0, metrics.bandwidth_mbps - self.bandwidth_probe_bw_cost_mbps),
            'loss_rate': metrics.loss_rate,
            'hop_count': len(path.as_sequence),
            'probe_time': self.current_time_slot * 900,
            'probe_type': 'full'
        }
        
        # Track both probe costs
        self.num_latency_probes += 1
        self.num_bandwidth_probes += 1
        self.latency_probe_time_ms += self.latency_probe_cost_ms
        self.bandwidth_probe_time_ms += self.bandwidth_probe_cost_ms
        self.total_probes += 2  # Both probe types
        self.total_probe_time_ms += (self.latency_probe_cost_ms + self.bandwidth_probe_cost_ms)
        self.total_probe_bandwidth_mbps += self.bandwidth_probe_bw_cost_mbps
        
        # Cache result
        self.probed_path_metrics[path_index] = probed_metrics
        
        return probed_metrics
    
    def probe_path(self, path_index: int) -> Dict:
        """
        Probe path based on configured probe type
        """
        if self.probe_type == 'latency_only':
            return self.probe_path_latency(path_index)
        elif self.probe_type == 'full':
            return self.probe_path_full(path_index)
        else:  # adaptive
            # Use heuristic: probe bandwidth only for promising paths
            latency_metrics = self.probe_path_latency(path_index)
            if latency_metrics['latency_ms'] < 100:  # Promising path
                return self.probe_path_full(path_index)
            return latency_metrics
    
    def probe_all_paths(self) -> List[Dict]:
        """
        Probe all available paths (for baseline methods)
        Uses configured probe type
        """
        all_metrics = []
        for i in range(len(self.available_paths)):
            metrics = self.probe_path(i)
            all_metrics.append(metrics)
        
        self.path_metrics = all_metrics
        return all_metrics
    
    def get_rl_exploration_paths(self, selected_path: int) -> List[int]:
        """
        Get additional paths for RL exploration
        
        Args:
            selected_path: The path selected by RL agent
            
        Returns:
            List of path indices to probe for exploration
        """
        exploration_paths = []
        
        # Always probe the selected path
        exploration_paths.append(selected_path)
        
        # Add exploration paths based on strategy
        available_indices = list(range(len(self.available_paths)))
        available_indices.remove(selected_path)
        
        if not available_indices:
            return exploration_paths
        
        # Strategy 1: Probe least recently probed paths
        path_ages = []
        current_time = self.current_time_slot * 900
        
        for idx in available_indices:
            path = self.available_paths[idx]
            path_key = hash(tuple(path.as_sequence))
            
            if path_key in self.path_probe_history:
                age = current_time - self.path_probe_history[path_key]
            else:
                age = float('inf')  # Never probed
            
            path_ages.append((idx, age))
        
        # Sort by age (oldest first)
        path_ages.sort(key=lambda x: x[1], reverse=True)
        
        # Take top exploration_budget paths
        for idx, _ in path_ages[:self.exploration_budget]:
            exploration_paths.append(idx)
        
        return exploration_paths
    
    def step_with_selective_probing(self, action: int, exploration_paths: Optional[List[int]] = None):
        """
        Step environment with selective probing for RL
        
        Args:
            action: Selected path index
            exploration_paths: Additional paths to probe for exploration
        """
        # Probe selected path and exploration paths
        if exploration_paths is None:
            exploration_paths = self.get_rl_exploration_paths(action)
        
        for path_idx in exploration_paths:
            if path_idx < len(self.available_paths):
                self.probe_path(path_idx)
                # Update probe history
                path = self.available_paths[path_idx]
                path_key = hash(tuple(path.as_sequence))
                self.path_probe_history[path_key] = self.current_time_slot * 900
        
        # Get actual metrics for selected path
        if action < len(self.available_paths) and action in self.probed_path_metrics:
            actual_metrics = self.probed_path_metrics[action]
        else:
            actual_metrics = self.probe_path(action)
        
        # Call parent step
        next_state, reward, terminated, truncated, info = super().step(action)
        
        # Add detailed probing stats to info
        info['probing_stats'] = {
            'num_latency_probes': self.num_latency_probes,
            'num_bandwidth_probes': self.num_bandwidth_probes,
            'total_probe_time_ms': self.total_probe_time_ms,
            'latency_probe_time_ms': self.latency_probe_time_ms,
            'bandwidth_probe_time_ms': self.bandwidth_probe_time_ms,
            'paths_probed': len(exploration_paths)
        }
        
        return next_state, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions (available path indices)
        
        Returns:
            List of valid action indices
        """
        return list(range(len(self.available_paths)))
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for valid actions
        
        Returns:
            Boolean mask where True indicates valid action
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[:len(self.available_paths)] = True
        return mask
    
    def get_path_metrics(self, path_index: int) -> Optional[Dict]:
        """
        Get metrics for a specific path (from cache if available)
        
        Args:
            path_index: Index of path
            
        Returns:
            Path metrics dict or None if invalid index
        """
        if path_index >= len(self.available_paths):
            return None
        
        if path_index in self.probed_path_metrics:
            return self.probed_path_metrics[path_index]
        
        # Return estimate if not probed
        return {
            'latency_ms': float('inf'),
            'bandwidth_mbps': 0,
            'loss_rate': 1.0,
            'hop_count': len(self.available_paths[path_index].as_sequence) if path_index < len(self.available_paths) else 0
        }
    
    def get_historical_bandwidth_estimate(self, path_index: int) -> Optional[float]:
        """
        Get bandwidth estimate from historical data or heuristics
        Used by RL to avoid expensive bandwidth probing
        """
        hist = self.get_historical_estimate(path_index)
        if hist and hist.get('bandwidth_mbps') is not None:
            return hist['bandwidth_mbps']
        
        # Heuristic: estimate based on hop count and path type
        if path_index < len(self.available_paths):
            path = self.available_paths[path_index]
            hop_count = len(path.as_sequence)
            
            # Simple heuristic: fewer hops = higher bandwidth
            if hop_count <= 2:
                return 1000.0  # 1 Gbps
            elif hop_count <= 4:
                return 500.0   # 500 Mbps
            elif hop_count <= 6:
                return 200.0   # 200 Mbps
            else:
                return 100.0   # 100 Mbps
        
        return None
    
    def get_probing_stats(self) -> Dict:
        """Get detailed statistics about probing overhead"""
        return {
            'total_probes': self.total_probes,
            'num_latency_probes': self.num_latency_probes,
            'num_bandwidth_probes': self.num_bandwidth_probes,
            'total_probe_time_ms': self.total_probe_time_ms,
            'latency_probe_time_ms': self.latency_probe_time_ms,
            'bandwidth_probe_time_ms': self.bandwidth_probe_time_ms,
            'total_probe_bandwidth_mbps': self.total_probe_bandwidth_mbps,
            'avg_probes_per_decision': self.total_probes / max(1, self.current_step),
            'probe_efficiency': self._calculate_probe_efficiency()
        }
    
    def _calculate_probe_efficiency(self) -> float:
        """
        Calculate probing efficiency metric
        Higher is better (more information per probe cost)
        """
        if self.total_probe_time_ms == 0:
            return 1.0
        
        # Efficiency = paths discovered / probe time
        paths_discovered = len(self.probed_path_metrics)
        efficiency = paths_discovered / (self.total_probe_time_ms / 1000)  # paths per second
        
        return min(1.0, efficiency / 10.0)  # Normalize to [0, 1]