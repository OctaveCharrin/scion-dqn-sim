"""
Realistic SCION path selection environment that accounts for probing overhead
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from .environment_fixed_source import SCIONPathSelectionEnvFixedSource as SCIONPathSelectionEnv

logger = logging.getLogger(__name__)


class RealisticSCIONPathSelectionEnv(SCIONPathSelectionEnv):
    """
    Enhanced environment that accounts for probing overhead and allows
    selective probing for RL agents
    """
    
    def __init__(self, *args, probe_overhead_ms: float = 10.0, 
                 probe_bandwidth_cost_mbps: float = 0.1, **kwargs):
        """
        Initialize realistic environment
        
        Args:
            probe_overhead_ms: Time overhead per probe (RTT)
            probe_bandwidth_cost_mbps: Bandwidth consumed per probe
        """
        super().__init__(*args, **kwargs)
        self.probe_overhead_ms = probe_overhead_ms
        self.probe_bandwidth_cost_mbps = probe_bandwidth_cost_mbps
        
        # Track probing costs
        self.total_probes = 0
        self.total_probe_time_ms = 0.0
        self.total_probe_bandwidth_mbps = 0.0
        
        # For RL: store historical path performance
        self.path_history = {}  # (src, dst, path_hash) -> recent_metrics
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        state, info = super().reset(seed=seed, options=options, **kwargs)

        # Don't probe paths yet - let the method decide
        self.unprobed_paths = self.available_paths.copy()
        self.probed_path_metrics = {}  # path_index -> metrics

        info = dict(info)
        info["unprobed_path_count"] = len(self.unprobed_paths)
        return state, info
    
    def probe_path(self, path_index: int) -> Dict:
        """
        Probe a specific path and return metrics
        Incurs realistic overhead
        """
        if path_index >= len(self.available_paths):
            return {
                'latency_ms': float('inf'),
                'bandwidth_mbps': 0,
                'loss_rate': 1.0,
                'hop_count': 0
            }
        
        # Check if already probed
        if path_index in self.probed_path_metrics:
            return self.probed_path_metrics[path_index]
        
        # Perform probe (with overhead)
        path = self.available_paths[path_index]
        adapted_path = self._adapt_path(path)
        
        # Get actual metrics
        metrics = self.pathprobe.probe(
            adapted_path,
            t_idx=self.current_time_slot,
            noisy=True
        )
        
        # Add probing overhead to latency
        probed_metrics = {
            'latency_ms': metrics.latency_ms + self.probe_overhead_ms,
            'bandwidth_mbps': max(0, metrics.bandwidth_mbps - self.probe_bandwidth_cost_mbps),
            'loss_rate': metrics.loss_rate,
            'hop_count': len(path.as_sequence),
            'probe_time': self.current_time_slot * 900  # When probed
        }
        
        # Track probing costs
        self.total_probes += 1
        self.total_probe_time_ms += self.probe_overhead_ms
        self.total_probe_bandwidth_mbps += self.probe_bandwidth_cost_mbps
        
        # Cache result
        self.probed_path_metrics[path_index] = probed_metrics
        
        return probed_metrics
    
    def probe_all_paths(self) -> List[Dict]:
        """
        Probe all available paths (for baseline methods)
        Returns metrics for all paths
        """
        all_metrics = []
        for i in range(len(self.available_paths)):
            metrics = self.probe_path(i)
            all_metrics.append(metrics)
        
        self.path_metrics = all_metrics
        return all_metrics
    
    def get_historical_estimate(self, path_index: int) -> Optional[Dict]:
        """
        Get historical performance estimate for a path (for RL)
        Returns None if no history available
        """
        if path_index >= len(self.available_paths):
            return None
            
        path = self.available_paths[path_index]
        path_key = (self.current_flow['src'], self.current_flow['dst'], 
                   hash(tuple(path.as_sequence)))
        
        if path_key in self.path_history:
            # Return average of recent observations
            recent = self.path_history[path_key]
            if len(recent) > 0:
                return {
                    'latency_ms': np.mean([m['latency_ms'] for m in recent]),
                    'bandwidth_mbps': np.mean([m['bandwidth_mbps'] for m in recent]),
                    'loss_rate': np.mean([m['loss_rate'] for m in recent]),
                    'hop_count': len(path.as_sequence),
                    'confidence': min(1.0, len(recent) / 10.0)  # Confidence score
                }
        
        return None
    
    def update_path_history(self, path_index: int, observed_metrics: Dict):
        """Update historical performance for a path"""
        if path_index >= len(self.available_paths):
            return
            
        path = self.available_paths[path_index]
        path_key = (self.current_flow['src'], self.current_flow['dst'],
                   hash(tuple(path.as_sequence)))
        
        if path_key not in self.path_history:
            self.path_history[path_key] = []
        
        # Keep last 20 observations
        self.path_history[path_key].append(observed_metrics)
        if len(self.path_history[path_key]) > 20:
            self.path_history[path_key].pop(0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step environment with selected action
        RL agents can use this without probing all paths first
        """
        # If path wasn't probed, probe it now (RL case)
        if action not in self.probed_path_metrics:
            actual_metrics = self.probe_path(action)
        else:
            actual_metrics = self.probed_path_metrics[action]
        
        # Call parent step
        next_state, reward, terminated, truncated, info = super().step(action)
        
        # Update historical data
        self.update_path_history(action, actual_metrics)
        
        # Add probing overhead info
        info['total_probes'] = self.total_probes
        info['probe_overhead_ms'] = self.total_probe_time_ms
        info['probe_bandwidth_cost_mbps'] = self.total_probe_bandwidth_mbps
        
        return next_state, reward, terminated, truncated, info
    
    def get_probing_stats(self) -> Dict:
        """Get statistics about probing overhead"""
        return {
            'total_probes': self.total_probes,
            'total_probe_time_ms': self.total_probe_time_ms,
            'total_probe_bandwidth_mbps': self.total_probe_bandwidth_mbps,
            'avg_probes_per_decision': self.total_probes / max(1, self.current_step)
        }


class BaselineSelector:
    """Base class for baseline selectors that need probing"""
    
    def __init__(self):
        self.requires_probing = True
    
    def select_path_with_probing(self, env: RealisticSCIONPathSelectionEnv) -> int:
        """Select path after probing all available paths"""
        # Probe all paths
        metrics = env.probe_all_paths()
        
        # Use specific selection logic
        return self._select_from_metrics(env.available_paths, metrics)
    
    def _select_from_metrics(self, paths: List[Any], metrics: List[Dict]) -> int:
        """Override in subclasses"""
        raise NotImplementedError


class RLSelector:
    """Base class for RL selectors that can work without full probing"""
    
    def __init__(self, agent):
        self.agent = agent
        self.requires_probing = False
    
    def select_path_without_probing(self, env: RealisticSCIONPathSelectionEnv, 
                                   state: np.ndarray) -> int:
        """
        Select path using learned policy without probing
        Can optionally use historical estimates
        """
        # Get historical estimates for all paths
        historical_estimates = []
        for i in range(len(env.available_paths)):
            est = env.get_historical_estimate(i)
            if est:
                historical_estimates.append(est)
            else:
                # No history - use default estimates based on hop count
                path = env.available_paths[i]
                historical_estimates.append({
                    'latency_ms': len(path.as_sequence) * 20,  # 20ms per hop estimate
                    'bandwidth_mbps': 1000 / len(path.as_sequence),  # Rough estimate
                    'loss_rate': 0.001 * len(path.as_sequence),  # 0.1% per hop
                    'hop_count': len(path.as_sequence),
                    'confidence': 0.0  # No confidence
                })
        
        # Let RL agent decide based on state (which includes historical info)
        valid_actions = list(range(len(env.available_paths)))
        action = self.agent.act(state, valid_actions)
        
        return action