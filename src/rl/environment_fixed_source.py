"""
Fixed-source Gymnasium environment for SCION path selection
Implements realistic deployment where agent runs at a specific AS
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
import pickle
import logging
from datetime import datetime

from ..path_services.pathfinder_v2 import PathFinderV2
from ..path_services.pathprobe import PathProbe
from .state_enhanced import EnhancedStateExtractor as StateExtractor
from .reward_with_probing import RewardCalculatorWithProbing as RewardCalculator

logger = logging.getLogger(__name__)

# Backward compatibility alias
FixedSourceSCIONPathSelectionEnv = None  # Will be set after class definition


class SCIONPathSelectionEnvFixedSource(gym.Env):
    """
    Gymnasium environment for SCION path selection with fixed source AS
    
    This environment models a realistic deployment where the path selection
    agent is deployed at a specific AS and makes decisions for outgoing traffic
    from that AS to various destinations.
    """

    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 topology_path: Path,
                 segments_path: Path,
                 link_table_path: Path,
                 metrics_path: Path,
                 metrics_shape: Tuple[int, int, int],
                 source_as: Optional[int] = None,
                 destination_ases: Optional[List[int]] = None,
                 config: Optional[Dict] = None):
        """
        Initialize SCION path selection environment with fixed source
        
        Args:
            topology_path: Path to topology pickle
            segments_path: Path to segments pickle
            link_table_path: Path to link table pickle
            metrics_path: Path to link metrics memmap
            metrics_shape: Shape of metrics array (time_slots, links, 3)
            source_as: Fixed source AS where agent is deployed (if None, will be set on first reset)
            destination_ases: List of possible destination ASes (if None, all ASes except source)
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or {}
        
        # Load topology
        with open(topology_path, 'rb') as f:
            self.topology = pickle.load(f)
        
        self.node_df = self.topology['nodes']
        self.edge_df = self.topology['edges']
        
        # Fixed source configuration
        self.fixed_source_as = source_as
        self.fixed_destination_ases = destination_ases
        self.source_as = None  # Will be set in reset
        self.destination_ases = None  # Will be set in reset
        
        # Initialize path services
        self.pathfinder = PathFinderV2(topology_path, segments_path, link_table_path)
        self.pathprobe = PathProbe(metrics_path, link_table_path, metrics_shape)
        
        # Environment parameters
        self.max_paths = self.config.get('max_paths', 10)
        self.time_slots = metrics_shape[0]
        self.current_time_slot = 0
        
        # State and action spaces
        self.state_extractor = StateExtractor(
            max_paths=self.max_paths,
            path_features=8,
            network_features=6,
            temporal_features=4
        )
        
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(self.state_extractor.state_dim,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Discrete(self.max_paths)
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(
            self.config.get('reward_config', {})
        )
        
        # Episode management
        self.episode_length = self.config.get('episode_length', 100)
        self.steps_in_episode = 0
        self.episode_stats = {
            'total_reward': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'avg_latency': [],
            'avg_bandwidth': [],
            'destinations_seen': set()
        }
        
        # Flow generation parameters
        self.flow_arrival_rate = self.config.get('flow_arrival_rate', 0.1)
        self.destination_popularity = {}  # Track destination popularity
        self.current_flow = None
        self.available_paths = []
        self.path_metrics = []
        
        # Path cache for efficiency (paths from source don't change)
        self.path_cache = {}
        
        logger.info(f"Initialized fixed-source SCION environment: "
                   f"{len(self.node_df)} ASes, "
                   f"{self.time_slots} time slots, "
                   f"max_paths={self.max_paths}, "
                   f"fixed_source={source_as}")
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.

        Pass ``source_as`` / ``dest_as`` either in ``options`` (Gymnasium style) or
        as keyword arguments for backward compatibility, e.g.::

            obs, info = env.reset(options={"source_as": 1, "dest_as": 2})
            obs, info = env.reset(source_as=1, dest_as=2)
        """
        opts = dict(options or {})
        for key in ("source_as", "dest_as"):
            if key in kwargs:
                opts[key] = kwargs[key]
        source_as = opts.get("source_as")
        dest_as = opts.get("dest_as")
        super().reset(seed=seed)

        # Set source AS (required on first reset if not provided in __init__)
        if source_as is not None:
            self.source_as = source_as
        elif self.fixed_source_as is not None:
            self.source_as = self.fixed_source_as
        elif self.source_as is None:
            raise ValueError("Source AS must be provided either in __init__ or first reset()")
        
        # Set destination ASes
        if self.fixed_destination_ases is not None:
            self.destination_ases = self.fixed_destination_ases
        else:
            # All ASes except source
            all_ases = self.node_df['as_id'].values
            self.destination_ases = [as_id for as_id in all_ases if as_id != self.source_as]
        
        # Initialize destination popularity (academic: use Zipf distribution)
        self._initialize_destination_popularity()
        
        # Reset time
        self.current_time_slot = np.random.randint(0, self.time_slots)
        self.steps_in_episode = 0
        
        # Reset episode statistics
        self.episode_stats = {
            'total_reward': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'avg_latency': [],
            'avg_bandwidth': [],
            'destinations_seen': set()
        }
        
        # Clear path cache periodically
        if np.random.random() < 0.1:  # 10% chance to clear cache
            self.path_cache.clear()
        
        # Generate initial flow
        self._generate_new_flow(dest_as=dest_as)
        
        # Get initial state
        state = self._get_state()
        info: Dict[str, Any] = {"source_as": self.source_as}
        return state, info

    def _initialize_destination_popularity(self):
        """Initialize destination popularity using Zipf distribution"""
        n_destinations = len(self.destination_ases)
        
        # Zipf distribution for realistic popularity
        s = 1.5  # Zipf parameter (higher = more skewed)
        weights = np.array([1 / (i ** s) for i in range(1, n_destinations + 1)])
        weights = weights / weights.sum()
        
        # Shuffle destinations and assign weights
        shuffled_dests = np.random.permutation(self.destination_ases)
        self.destination_popularity = {
            dest: weight for dest, weight in zip(shuffled_dests, weights)
        }
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute path selection action (Gymnasium step API)."""
        # Validate action
        if action >= len(self.available_paths):
            # Invalid action - use shortest path as fallback
            action = 0
            valid_action = False
        else:
            valid_action = True
        
        # Get selected path
        if self.available_paths and action < len(self.available_paths):
            selected_path = self.available_paths[action]
            selected_metrics = self.path_metrics[action]
        else:
            selected_path = None
            selected_metrics = {
                'latency_ms': float('inf'),
                'bandwidth_mbps': 0,
                'loss_rate': 1.0
            }
        
        # Calculate reward
        flow_requirements = self._get_flow_requirements()
        reward = self.reward_calculator.calculate_reward(
            selected_metrics,
            flow_requirements,
            valid_action and selected_path is not None
        )
        
        # Update episode statistics
        self.episode_stats['total_reward'] += reward
        if reward > 0:
            self.episode_stats['successful_selections'] += 1
            self.episode_stats['avg_latency'].append(selected_metrics['latency_ms'])
            self.episode_stats['avg_bandwidth'].append(selected_metrics['bandwidth_mbps'])
        else:
            self.episode_stats['failed_selections'] += 1
        
        # Track destination
        self.episode_stats['destinations_seen'].add(self.current_flow['dst'])
        
        # Advance time
        self.current_time_slot = (self.current_time_slot + 1) % self.time_slots
        self.steps_in_episode += 1
        
        # Check if episode is done (no truncation in this env)
        terminated = self.steps_in_episode >= self.episode_length
        truncated = False

        # Generate new flow for next step
        if not terminated:
            self._generate_new_flow()
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.observation_space.shape[0])
        
        # Prepare info
        info: Dict[str, Any] = {
            'valid_action': valid_action,
            'selected_path': selected_path,
            'path_metrics': selected_metrics,
            'flow': self.current_flow,
            'available_paths': len(self.available_paths),
            'time_slot': self.current_time_slot,
            'source_as': self.source_as
        }
        
        if terminated:
            info['episode_stats'] = self._get_episode_summary()
        
        return next_state, reward, terminated, truncated, info
    
    def _generate_new_flow(self, dest_as: Optional[int] = None):
        """
        Generate a new flow request from fixed source
        
        Args:
            dest_as: Specific destination AS (optional)
        """
        # Source is always fixed
        src = self.source_as
        
        # Select destination
        if dest_as is not None and dest_as in self.destination_ases:
            dst = dest_as
        else:
            # Use popularity-based selection (academic best practice)
            if self.destination_popularity:
                dests = list(self.destination_popularity.keys())
                weights = list(self.destination_popularity.values())
                dst = np.random.choice(dests, p=weights)
            else:
                # Fallback to uniform random
                dst = np.random.choice(self.destination_ases)
        
        # Generate flow characteristics
        # Academic: use realistic traffic models
        traffic_class = np.random.choice(['mice', 'elephant'], p=[0.8, 0.2])
        
        if traffic_class == 'mice':
            # Small flows (web browsing, API calls)
            size_mb = np.random.exponential(1)  # Mean 1 MB
            priority = np.random.choice(['low', 'medium'], p=[0.7, 0.3])
        else:
            # Large flows (file transfers, backups)
            size_mb = np.random.lognormal(3.0, 1.5)  # Log-normal distribution
            priority = np.random.choice(['medium', 'high'], p=[0.6, 0.4])
        
        self.current_flow = {
            'src': int(src),
            'dst': int(dst),
            'size_mb': size_mb,
            'priority': priority,
            'traffic_class': traffic_class
        }
        
        # Look up available paths (with caching for efficiency)
        cache_key = (src, dst)
        if cache_key in self.path_cache:
            self.available_paths = self.path_cache[cache_key]
        else:
            self.available_paths = self.pathfinder.get_paths(
                self.current_flow['src'],
                self.current_flow['dst'],
                k=self.max_paths
            )
            self.path_cache[cache_key] = self.available_paths
        
        # Probe path metrics
        self.path_metrics = []
        for path in self.available_paths:
            try:
                # Create path adapter
                adapted_path = self._adapt_path(path)
                metrics = self.pathprobe.probe(
                    adapted_path,
                    t_idx=self.current_time_slot,
                    noisy=True  # Add realistic noise
                )
                
                self.path_metrics.append({
                    'latency_ms': metrics.latency_ms,
                    'bandwidth_mbps': metrics.bandwidth_mbps,
                    'loss_rate': metrics.loss_rate,
                    'hop_count': len(path.as_sequence)
                })
            except Exception as e:
                logger.warning(f"Failed to probe path: {e}")
                self.path_metrics.append({
                    'latency_ms': float('inf'),
                    'bandwidth_mbps': 0,
                    'loss_rate': 1.0,
                    'hop_count': len(path.as_sequence)
                })
    
    def _adapt_path(self, path):
        """Adapt PathFinderV2 path to PathProbe format"""
        class PathAdapter:
            def __init__(self, p):
                self.src = p.src
                self.dst = p.dst
                self.hops = p.as_sequence
                self.interfaces = tuple(
                    (h.egress_if, nh.ingress_if)
                    for h, nh in zip(p.hops[:-1], p.hops[1:])
                )
                self.segment_types = (p.path_type,)
                self.total_hops = p.total_hops
        
        return PathAdapter(path)
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state vector from the perspective of source AS
        
        Academic insight: State should only include information
        observable from the source AS location
        """
        # Get network state visible from source
        network_state = self._get_local_network_state()
        
        # Extract state features
        state = self.state_extractor.extract_state(
            src_as=self.current_flow['src'],
            dst_as=self.current_flow['dst'],
            paths=self.available_paths,
            path_metrics=self.path_metrics,
            network_state=network_state,
            network_graph=None,  # Not using graph features for now
            current_time=self.current_time_slot * 900  # Convert to seconds (15 min slots)
        )
        
        return state
    
    def _get_local_network_state(self) -> Dict:
        """
        Get network state observable from source AS
        
        Academic: Only include information that would be available
        to an agent deployed at the source AS
        """
        # Get direct neighbors of source AS
        neighbors = set()
        for _, edge in self.edge_df.iterrows():
            # Handle both column naming conventions
            from_as = edge.get('from_as', edge.get('u'))
            to_as = edge.get('to_as', edge.get('v'))
            
            if from_as == self.source_as:
                neighbors.add(to_as)
            elif to_as == self.source_as:
                neighbors.add(from_as)
        
        # Aggregate statistics visible from source
        total_bandwidth = 0
        avg_latency = 0
        link_count = 0
        
        for _, edge in self.edge_df.iterrows():
            # Handle both column naming conventions
            from_as = edge.get('from_as', edge.get('u'))
            to_as = edge.get('to_as', edge.get('v'))
            
            # Only consider links visible from source (direct or one-hop)
            if from_as == self.source_as or to_as == self.source_as:
                total_bandwidth += edge['bandwidth']
                # Calculate latency from distance (speed of light in fiber: ~200km/ms)
                latency_ms = edge.get('latency', edge.get('dist_km', 100) / 200.0)
                avg_latency += latency_ms
                link_count += 1
            elif from_as in neighbors or to_as in neighbors:
                # One-hop visibility (academic: BGP announcements)
                total_bandwidth += edge['bandwidth'] * 0.5  # Partial visibility
                latency_ms = edge.get('latency', edge.get('dist_km', 100) / 200.0)
                avg_latency += latency_ms
                link_count += 1
        
        if link_count > 0:
            avg_latency /= link_count
        
        # Get current network load (if available from metrics)
        current_load = self._estimate_local_load()
        
        return {
            'total_bandwidth': total_bandwidth,
            'avg_latency': avg_latency,
            'link_count': link_count,
            'neighbor_count': len(neighbors),
            'current_load': current_load,
            'source_as_degree': len(neighbors)
        }
    
    def _estimate_local_load(self) -> float:
        """Estimate network load visible from source AS"""
        # Academic: Use exponential smoothing of recent observations
        # In practice, this would come from local monitoring
        base_load = 0.5  # Base utilization
        time_factor = np.sin(2 * np.pi * self.current_time_slot / self.time_slots) * 0.3
        noise = np.random.normal(0, 0.05)
        
        return np.clip(base_load + time_factor + noise, 0, 1)
    
    def _get_time_features(self) -> np.ndarray:
        """Get temporal features"""
        # Time of day (normalized)
        time_of_day = self.current_time_slot / self.time_slots
        
        # Sine/cosine encoding for cyclical nature
        sin_time = np.sin(2 * np.pi * time_of_day)
        cos_time = np.cos(2 * np.pi * time_of_day)
        
        # Day of week approximation (if time_slots represents hours)
        if self.time_slots >= 168:  # At least a week
            day_of_week = (self.current_time_slot // 24) % 7 / 7
        else:
            day_of_week = 0.5
        
        return np.array([time_of_day, sin_time, cos_time, day_of_week])
    
    def _get_flow_requirements(self) -> Dict:
        """Get flow requirements based on priority and size"""
        requirements = {
            'max_latency': 1000,  # ms
            'min_bandwidth': 1,   # Mbps
            'max_loss': 0.01      # 1%
        }
        
        # Adjust based on priority
        if self.current_flow['priority'] == 'high':
            requirements['max_latency'] = 50
            requirements['min_bandwidth'] = 100
            requirements['max_loss'] = 0.001
        elif self.current_flow['priority'] == 'medium':
            requirements['max_latency'] = 200
            requirements['min_bandwidth'] = 10
            requirements['max_loss'] = 0.005
        
        # Adjust based on flow size
        if self.current_flow['size_mb'] > 100:
            # Large flows care more about bandwidth
            requirements['min_bandwidth'] *= 2
        
        return requirements
    
    def _get_episode_summary(self) -> Dict:
        """Get episode summary statistics"""
        summary = {
            'total_reward': self.episode_stats['total_reward'],
            'success_rate': self.episode_stats['successful_selections'] / max(1, self.steps_in_episode),
            'avg_latency': np.mean(self.episode_stats['avg_latency']) if self.episode_stats['avg_latency'] else 0,
            'avg_bandwidth': np.mean(self.episode_stats['avg_bandwidth']) if self.episode_stats['avg_bandwidth'] else 0,
            'unique_destinations': len(self.episode_stats['destinations_seen']),
            'destination_coverage': len(self.episode_stats['destinations_seen']) / len(self.destination_ases)
        }
        
        return summary
    
    def set_source_destination_config(self, source_as: int, destination_ases: List[int]):
        """
        Set fixed source and destination configuration
        
        Args:
            source_as: Source AS where agent is deployed
            destination_ases: List of possible destination ASes
        """
        self.fixed_source_as = source_as
        self.fixed_destination_ases = destination_ases
        self.source_as = source_as
        self.destination_ases = destination_ases
        
        # Clear cache when configuration changes
        self.path_cache.clear()
        
        logger.info(f"Updated fixed source configuration: "
                   f"source={source_as}, destinations={len(destination_ases)}")


# Backward compatibility: create alias
SCIONPathSelectionEnv = SCIONPathSelectionEnvFixedSource
# Set backward compatibility alias
FixedSourceSCIONPathSelectionEnv = SCIONPathSelectionEnvFixedSource
