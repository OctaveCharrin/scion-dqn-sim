"""
Legacy Gym-environment reward calculator (probing-aware).

The evaluation pipeline (steps 04–05) uses
:class:`~src.simulation.evaluation_env.EvaluationPathSelectionEnv` and
:meth:`~src.simulation.evaluation_env.EvaluationPathSelectionEnv.compute_reward`
instead. This module remains for ``src.rl.environment_*`` Gym flows.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RewardCalculatorWithProbing:
    """Calculate rewards including probing overhead costs"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reward calculator with probing costs
        
        Args:
            config: Reward configuration including probing costs
        """
        config = config or {}
        
        # Base reward weights
        self.throughput_weight = config.get('throughput_weight', 0.3)
        self.latency_weight = config.get('latency_weight', 0.25)
        self.reliability_weight = config.get('reliability_weight', 0.15)
        self.cost_weight = config.get('cost_weight', 0.1)
        self.probing_weight = config.get('probing_weight', 0.2)  # NEW: weight for probing cost
        
        # Normalization parameters
        self.max_throughput = config.get('max_throughput_mbps', 10000)
        self.max_latency = config.get('max_latency_ms', 1000)
        self.target_latency = config.get('target_latency_ms', 50)
        
        # Probing cost parameters
        self.latency_probe_cost_ms = config.get('latency_probe_cost_ms', 10.0)
        self.bandwidth_probe_cost_ms = config.get('bandwidth_probe_cost_ms', 100.0)
        self.bandwidth_probe_bw_cost_mbps = config.get('bandwidth_probe_bw_cost_mbps', 10.0)
        self.max_acceptable_probe_time_ms = config.get('max_acceptable_probe_time_ms', 500.0)
        
        # Penalty factors
        self.failure_penalty = config.get('failure_penalty', -1.0)
        self.sla_violation_penalty = config.get('sla_violation_penalty', -0.5)
        
        logger.info(f"Initialized reward calculator with probing costs: "
                   f"latency_probe={self.latency_probe_cost_ms}ms, "
                   f"bandwidth_probe={self.bandwidth_probe_cost_ms}ms")
    
    def calculate_reward(self,
                        path_metrics: Dict,
                        flow_requirements: Optional[Dict] = None,
                        action_valid: bool = True,
                        probing_stats: Optional[Dict] = None) -> float:
        """
        Calculate reward for path selection including probing costs
        
        Args:
            path_metrics: Metrics from path probe
            flow_requirements: Optional flow requirements
            action_valid: Whether the selected action was valid
            probing_stats: Statistics about probing performed
                - num_latency_probes: Number of latency probes
                - num_bandwidth_probes: Number of bandwidth probes
                - total_probe_time_ms: Total time spent probing
            
        Returns:
            Reward value
        """
        if not action_valid:
            return self.failure_penalty
        
        # Extract metrics
        latency = path_metrics.get('latency_ms', float('inf'))
        bandwidth = path_metrics.get('bandwidth_mbps', 0)
        loss_rate = path_metrics.get('loss_rate', 1.0)
        
        # Check for path failure
        if bandwidth == 0 or loss_rate > 0.5 or latency == float('inf'):
            return self.failure_penalty
        
        # Calculate base component rewards
        throughput_reward = self._calculate_throughput_reward(bandwidth, flow_requirements)
        latency_reward = self._calculate_latency_reward(latency, flow_requirements)
        reliability_reward = self._calculate_reliability_reward(loss_rate)
        cost_reward = self._calculate_cost_reward(path_metrics)
        
        # Calculate probing cost penalty
        probing_reward = self._calculate_probing_reward(probing_stats)
        
        # Combine rewards
        total_reward = (
            self.throughput_weight * throughput_reward +
            self.latency_weight * latency_reward +
            self.reliability_weight * reliability_reward +
            self.cost_weight * cost_reward +
            self.probing_weight * probing_reward
        )
        
        # Apply SLA violation penalties if requirements exist
        if flow_requirements:
            sla_penalty = self._check_sla_violations(
                path_metrics, flow_requirements
            )
            total_reward += sla_penalty
        
        # Ensure reward is in [-1, 1] range
        return np.clip(total_reward, -1.0, 1.0)
    
    def _calculate_probing_reward(self, probing_stats: Optional[Dict]) -> float:
        """
        Calculate probing cost component of reward
        Higher probing overhead = lower reward
        """
        if not probing_stats:
            # No probing info - assume minimal probing (good)
            return 1.0
        
        num_latency_probes = probing_stats.get('num_latency_probes', 0)
        num_bandwidth_probes = probing_stats.get('num_bandwidth_probes', 0)
        total_probe_time = probing_stats.get('total_probe_time_ms', 0)
        
        # Calculate expected probe time
        expected_time = (
            num_latency_probes * self.latency_probe_cost_ms +
            num_bandwidth_probes * self.bandwidth_probe_cost_ms
        )
        
        # Normalize to [-1, 1]
        if total_probe_time == 0:
            # No probing - excellent!
            return 1.0
        elif total_probe_time <= 50:
            # Minimal probing (1-2 paths)
            return 0.8
        elif total_probe_time <= 100:
            # Moderate probing (few paths)
            return 0.5
        elif total_probe_time <= 200:
            # Significant probing
            return 0.0
        elif total_probe_time <= self.max_acceptable_probe_time_ms:
            # Heavy probing but acceptable
            return -0.5
        else:
            # Excessive probing
            return -1.0
    
    def _calculate_throughput_reward(self, 
                                   bandwidth_mbps: float,
                                   requirements: Optional[Dict]) -> float:
        """Calculate throughput component of reward"""
        if requirements and 'min_bandwidth_mbps' in requirements:
            min_bw = requirements['min_bandwidth_mbps']
            if bandwidth_mbps < min_bw:
                # Penalty for not meeting requirement
                return -1.0 * (1 - bandwidth_mbps / min_bw)
            else:
                # Reward for exceeding requirement
                excess = (bandwidth_mbps - min_bw) / min_bw
                return min(1.0, excess / 2)  # Diminishing returns
        else:
            # No specific requirement - normalize
            return min(1.0, bandwidth_mbps / self.max_throughput)
    
    def _calculate_latency_reward(self,
                                latency_ms: float,
                                requirements: Optional[Dict]) -> float:
        """Calculate latency component of reward"""
        if requirements and 'max_latency_ms' in requirements:
            max_lat = requirements['max_latency_ms']
            if latency_ms > max_lat:
                # Penalty for exceeding requirement
                return -1.0 * min(1.0, (latency_ms - max_lat) / max_lat)
            else:
                # Reward for being under requirement
                return (max_lat - latency_ms) / max_lat
        else:
            # No specific requirement - use target latency
            if latency_ms <= self.target_latency:
                return 1.0
            elif latency_ms >= self.max_latency:
                return -1.0
            else:
                # Linear interpolation
                return 1.0 - 2.0 * (latency_ms - self.target_latency) / (
                    self.max_latency - self.target_latency
                )
    
    def _calculate_reliability_reward(self, loss_rate: float) -> float:
        """Calculate reliability component of reward"""
        # Map loss rate to reward
        if loss_rate == 0:
            return 1.0
        elif loss_rate < 0.001:  # < 0.1%
            return 0.9
        elif loss_rate < 0.01:   # < 1%
            return 0.5
        elif loss_rate < 0.05:   # < 5%
            return 0.0
        elif loss_rate < 0.1:    # < 10%
            return -0.5
        else:
            return -1.0
    
    def _calculate_cost_reward(self, path_metrics: Dict) -> float:
        """Calculate cost component of reward"""
        # Simple cost model based on hop count
        hop_count = path_metrics.get('hop_count', 10)
        
        # Fewer hops = lower cost = higher reward
        if hop_count <= 2:
            return 1.0
        elif hop_count <= 4:
            return 0.5
        elif hop_count <= 6:
            return 0.0
        elif hop_count <= 8:
            return -0.5
        else:
            return -1.0
    
    def _check_sla_violations(self,
                            path_metrics: Dict,
                            requirements: Dict) -> float:
        """Check for SLA violations and return penalty"""
        penalty = 0.0
        
        # Bandwidth violation
        if 'min_bandwidth_mbps' in requirements:
            if path_metrics.get('bandwidth_mbps', 0) < requirements['min_bandwidth_mbps']:
                penalty += self.sla_violation_penalty
        
        # Latency violation
        if 'max_latency_ms' in requirements:
            if path_metrics.get('latency_ms', float('inf')) > requirements['max_latency_ms']:
                penalty += self.sla_violation_penalty
        
        # Loss rate violation
        if 'max_loss_rate' in requirements:
            if path_metrics.get('loss_rate', 1.0) > requirements['max_loss_rate']:
                penalty += self.sla_violation_penalty
        
        return penalty