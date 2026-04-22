"""
SCION Path Selection Environment for Evaluation Pipeline
Compatible with evaluation data formats
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EvaluationEnv(gym.Env):
    """
    Gymnasium environment for SCION path selection evaluation
    Works directly with evaluation pipeline data formats
    """

    metadata = {"render_modes": ["human"]}
    
    def __init__(self,
                 topology: Any,
                 segment_store: Any,
                 flow_data_path: str,
                 source_as: int,
                 destination_ases: List[int],
                 training_days: int = 14,
                 episode_duration: int = 1440,  # 1 day in minutes
                 max_paths: int = 10,
                 reward_config: Optional[Dict] = None):
        """
        Initialize environment
        
        Args:
            topology: Topology object (dict or NetworkX)
            segment_store: Segment store object
            flow_data_path: Path to flow data NPZ file
            source_as: Fixed source AS
            destination_ases: List of destination ASes
            training_days: Number of days to use for training
            episode_duration: Duration of each episode in minutes
            max_paths: Maximum number of paths to consider
            reward_config: Reward function configuration
        """
        super().__init__()
        
        self.topology = topology
        self.segment_store = segment_store
        self.source_as = source_as
        self.destination_ases = destination_ases
        self.training_days = training_days
        self.episode_duration = episode_duration
        self.max_paths = max_paths
        self.reward_config = reward_config or {
            'success_weight': 1.0,
            'latency_weight': 0.3,
            'throughput_weight': 0.2,
            'loss_weight': 0.1
        }
        
        # Load flow data
        self._load_flow_data(flow_data_path)
        
        # Initialize path cache
        self.path_cache = {}
        self._build_path_cache()
        
        # State configuration
        self.path_features = 8  # hops, latency, bandwidth, loss, freshness, diversity, load, reliability
        self.network_features = 6  # avg_utilization, max_utilization, total_flows, failed_flows, avg_latency, path_availability
        self.temporal_features = 4  # hour_sin, hour_cos, day_sin, day_cos
        self.state_dim = self.max_paths * self.path_features + self.network_features + self.temporal_features
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(self.max_paths)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.current_time = 0
        self.episode_flows = []
        self.network_state = {
            'link_utilization': {},
            'path_usage': {},
            'recent_failures': deque(maxlen=100)
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_successes = []
    
    def _load_flow_data(self, flow_data_path: str):
        """Load and preprocess flow data"""
        data = np.load(flow_data_path)
        
        # Filter for training period
        training_minutes = self.training_days * 24 * 60
        mask = data['start_time'] < training_minutes
        
        # Filter for flows from source AS
        mask &= data['source'] == self.source_as
        
        # Filter for flows to destination ASes
        dest_mask = np.zeros(len(data['source']), dtype=bool)
        for dest in self.destination_ases:
            dest_mask |= (data['destination'] == dest)
        mask &= dest_mask
        
        # Store filtered flow data
        self.flow_data = {
            'flow_id': data['flow_id'][mask],
            'source': data['source'][mask],
            'destination': data['destination'][mask],
            'start_time': data['start_time'][mask],
            'size_bytes': data['size_bytes'][mask],
            'requested_rate_mbps': data['requested_rate_mbps'][mask],
            'status': data['status'][mask]
        }
        
        self.n_flows = len(self.flow_data['flow_id'])
        logger.info(f"Loaded {self.n_flows} flows for training")
    
    def _build_path_cache(self):
        """Build cache of available paths between source and destinations"""
        for dest in self.destination_ases:
            paths = self._find_paths(self.source_as, dest)
            if paths:
                self.path_cache[(self.source_as, dest)] = paths[:self.max_paths]
            else:
                # Create dummy path if no real path exists
                self.path_cache[(self.source_as, dest)] = [{
                    'hops': [self.source_as, dest],
                    'latency': 50.0,
                    'bandwidth': 100.0,
                    'loss': 0.01
                }]
    
    def _find_paths(self, src: int, dst: int) -> List[Dict]:
        """Find paths between source and destination using segment store"""
        paths = []
        
        # Simple path finding logic (can be enhanced)
        if isinstance(self.segment_store, dict):
            # Try to find up-down paths through core
            up_segments = self.segment_store.get('up', {}).get(src, [])
            down_segments = self.segment_store.get('down', {}).get(dst, [])
            
            # Create paths by combining segments
            for up_seg in up_segments[:3]:  # Limit for performance
                for down_seg in down_segments[:3]:
                    if up_seg.get('core_as') == down_seg.get('core_as'):
                        path = {
                            'hops': up_seg.get('hops', []) + down_seg.get('hops', [])[1:],
                            'latency': up_seg.get('latency', 20) + down_seg.get('latency', 20),
                            'bandwidth': min(up_seg.get('bandwidth', 1000), down_seg.get('bandwidth', 1000)),
                            'loss': up_seg.get('loss', 0.001) + down_seg.get('loss', 0.001)
                        }
                        paths.append(path)
        
        # If no paths found, create direct path
        if not paths:
            paths.append({
                'hops': [src, dst],
                'latency': 30.0,
                'bandwidth': 1000.0,
                'loss': 0.001
            })
        
        return paths
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_time = np.random.randint(0, self.training_days * 24 * 60 - self.episode_duration)
        self.episode_flows = []
        self.episode_rewards = []
        self.episode_successes = []
        
        # Reset network state
        self.network_state = {
            'link_utilization': {},
            'path_usage': {},
            'recent_failures': deque(maxlen=100)
        }
        
        # Get initial state
        obs = self._get_state()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return results (Gymnasium step API)."""
        # Get current flow
        flow_idx = self._get_current_flow_index()
        if flow_idx is None:
            # No flow at this time, skip
            reward = 0
            info = {'no_flow': True}
        else:
            flow = self._get_flow(flow_idx)
            dest = flow['destination']
            
            # Get available paths
            paths = self.path_cache.get((self.source_as, dest), [])
            
            # Select path based on action
            if action < len(paths):
                selected_path = paths[action]
                success = self._simulate_flow_on_path(flow, selected_path)
            else:
                # Invalid action, use first available path
                selected_path = paths[0] if paths else None
                success = False
            
            # Calculate reward
            reward = self._calculate_reward(flow, selected_path, success)
            self.episode_rewards.append(reward)
            self.episode_successes.append(success)
            
            # Update network state
            self._update_network_state(flow, selected_path, success)
            
            info = {
                'flow_id': flow['flow_id'],
                'success': success,
                'selected_path': selected_path,
                'destination': dest
            }
        
        # Update time
        self.current_step += 1
        self.current_time += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.episode_duration
        truncated = False

        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, terminated, truncated, info
    
    def _get_current_flow_index(self) -> Optional[int]:
        """Get index of flow starting at current time"""
        # Find flows starting at current time
        mask = self.flow_data['start_time'] == self.current_time
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            return indices[0]
        return None
    
    def _get_flow(self, idx: int) -> Dict:
        """Get flow data by index"""
        return {
            'flow_id': self.flow_data['flow_id'][idx],
            'source': self.flow_data['source'][idx],
            'destination': self.flow_data['destination'][idx],
            'start_time': self.flow_data['start_time'][idx],
            'size_bytes': self.flow_data['size_bytes'][idx],
            'requested_rate_mbps': self.flow_data['requested_rate_mbps'][idx]
        }
    
    def _simulate_flow_on_path(self, flow: Dict, path: Dict) -> bool:
        """Simulate sending flow on selected path"""
        # Simple simulation based on path characteristics
        if path is None:
            return False
        
        # Success probability based on path quality
        base_success_rate = 0.95
        
        # Reduce success rate based on path characteristics
        if path['loss'] > 0.02:
            base_success_rate -= 0.1
        if path['latency'] > 100:
            base_success_rate -= 0.05
        if path['bandwidth'] < flow['requested_rate_mbps']:
            base_success_rate -= 0.15
        
        # Add some randomness
        success = np.random.random() < base_success_rate
        
        return success
    
    def _calculate_reward(self, flow: Dict, path: Dict, success: bool) -> float:
        """Calculate reward for path selection"""
        if path is None:
            return -1.0
        
        reward = 0.0
        
        # Success/failure component
        if success:
            reward += self.reward_config['success_weight'] * 1.0
        else:
            reward -= self.reward_config['success_weight'] * 1.0
        
        # Latency component (normalized)
        normalized_latency = np.clip(path['latency'] / 100.0, 0, 1)
        reward -= self.reward_config['latency_weight'] * normalized_latency
        
        # Throughput component (satisfaction ratio)
        throughput_ratio = min(path['bandwidth'] / flow['requested_rate_mbps'], 1.0)
        reward += self.reward_config['throughput_weight'] * throughput_ratio
        
        # Loss component
        reward -= self.reward_config['loss_weight'] * path['loss'] * 10
        
        return reward
    
    def _update_network_state(self, flow: Dict, path: Dict, success: bool):
        """Update network state based on flow result"""
        if path is None:
            return
        
        # Update path usage
        path_key = tuple(path['hops'])
        self.network_state['path_usage'][path_key] = \
            self.network_state['path_usage'].get(path_key, 0) + 1
        
        # Update link utilization (simplified)
        for i in range(len(path['hops']) - 1):
            link = (path['hops'][i], path['hops'][i+1])
            self.network_state['link_utilization'][link] = \
                self.network_state['link_utilization'].get(link, 0) + flow['requested_rate_mbps']
        
        # Track failures
        if not success:
            self.network_state['recent_failures'].append({
                'time': self.current_time,
                'path': path_key,
                'destination': flow['destination']
            })
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Get next flow destination (if any)
        next_flow_idx = self._get_current_flow_index()
        if next_flow_idx is not None:
            next_flow = self._get_flow(next_flow_idx)
            dest = next_flow['destination']
            paths = self.path_cache.get((self.source_as, dest), [])
        else:
            # Use random destination for state
            dest = np.random.choice(self.destination_ases)
            paths = self.path_cache.get((self.source_as, dest), [])
        
        # Path features
        for i in range(self.max_paths):
            if i < len(paths):
                path = paths[i]
                # Extract path features
                features = [
                    len(path['hops']) / 10.0,  # Normalized hop count
                    path['latency'] / 100.0,    # Normalized latency
                    path['bandwidth'] / 1000.0,  # Normalized bandwidth
                    path['loss'] * 100,          # Loss percentage
                    0.5,                         # Freshness (mock)
                    0.5,                         # Diversity (mock)
                    0.3,                         # Load (mock)
                    0.8                          # Reliability (mock)
                ]
            else:
                # Padding for missing paths
                features = [0.0] * self.path_features
            
            state.extend(features)
        
        # Network features
        total_utilization = sum(self.network_state['link_utilization'].values())
        max_utilization = max(self.network_state['link_utilization'].values()) if self.network_state['link_utilization'] else 0
        
        network_features = [
            total_utilization / 10000.0,  # Normalized total utilization
            max_utilization / 1000.0,     # Normalized max utilization
            len(self.episode_flows) / 100.0,  # Normalized flow count
            len(self.network_state['recent_failures']) / 100.0,  # Failure rate
            0.5,  # Average latency (mock)
            len(paths) / self.max_paths  # Path availability
        ]
        state.extend(network_features)
        
        # Temporal features
        hour = (self.current_time // 60) % 24
        day = (self.current_time // (60 * 24)) % 7
        
        temporal_features = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7),
            np.cos(2 * np.pi * day / 7)
        ]
        state.extend(temporal_features)
        
        return np.array(state, dtype=np.float32)
    
    def render(self):
        """Print simple text diagnostics."""
        print(f"Step: {self.current_step}, Time: {self.current_time}")
        print(f"Recent success rate: {np.mean(self.episode_successes[-10:]):.2%}")
        print(f"Average reward: {np.mean(self.episode_rewards[-10:]):.2f}")
        return None
    
    def close(self):
        """Clean up environment"""
        pass