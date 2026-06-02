"""
Capacity and delay builder for link annotation

Assigns capacities and calculates propagation delays for all links.
Uses GraphBLAS for efficient sparse matrix operations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pickle

try:
    import pygraphblas as gb
except ImportError:
    gb = None
    print("Warning: pygraphblas not available, using NumPy fallback")


class CapacityDelayBuilder:
    """Build capacity and delay annotations for links"""
    
    # Capacity tiers in Gbps
    CAPACITY_TIERS = [1, 10, 40, 100]
    
    # Physical constants
    PROP_SPEED_KM_PER_MS = 204.08  # Speed of light in fiber (c/1.49)
    BASE_PROCESSING_DELAY_MS = 0.10  # Fixed processing delay
    
    def __init__(self, 
                 capacity_mu: float = 40.0,
                 capacity_sigma: float = 1.0,
                 min_leaf_capacity: float = 1.0):
        """
        Args:
            capacity_mu: Mean of log-normal distribution for capacity (Gbps)
            capacity_sigma: Std dev of log-normal distribution
            min_leaf_capacity: Minimum capacity for leaf-parent links (Gbps)
        """
        self.capacity_mu = np.log(capacity_mu)
        self.capacity_sigma = capacity_sigma
        self.min_leaf_capacity = min_leaf_capacity
        
    def annotate(self, topology_path: Path, output_path: Path) -> pd.DataFrame:
        """
        Annotate topology with capacities and delays
        
        Args:
            topology_path: Path to topology pickle file
            output_path: Where to save link table
            
        Returns:
            DataFrame with link annotations
        """
        # Load topology
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            
        node_df = topology['nodes']
        edge_df = topology['edges']
        
        # Assign capacities
        capacities = self._assign_capacities(edge_df, node_df)
        
        # Calculate propagation delays
        prop_delays = self._calculate_prop_delays(edge_df)
        
        # Build link table
        link_table = pd.DataFrame({
            'u': edge_df['u'],
            'v': edge_df['v'],
            'u_if': edge_df['u_if'],
            'v_if': edge_df['v_if'],
            'type': edge_df['type'],
            'capacity_gbps': capacities,
            'prop_delay_ms': prop_delays,
            'dist_km': edge_df['dist_km']
        })
        
        # Add reverse links (SCION links are bidirectional)
        reverse_links = link_table.copy()
        reverse_links['u'] = link_table['v']
        reverse_links['v'] = link_table['u']
        reverse_links['u_if'] = link_table['v_if']
        reverse_links['v_if'] = link_table['u_if']
        
        # Combine forward and reverse
        link_table = pd.concat([link_table, reverse_links], ignore_index=True)
        
        # Save to pickle (using parquet requires pyarrow)
        link_table.to_pickle(output_path)
        
        return link_table
    
    def _assign_capacities(self, edge_df: pd.DataFrame, 
                          node_df: pd.DataFrame) -> np.ndarray:
        """Assign capacity tiers based on link types and node roles"""
        capacities = []
        
        # Create role lookup
        node_roles = {row['as_id']: row['role'] 
                     for _, row in node_df.iterrows()}
        
        for _, edge in edge_df.iterrows():
            u_role = node_roles[edge['u']]
            v_role = node_roles[edge['v']]
            link_type = edge['type']
            
            # Draw base capacity from log-normal
            base_cap = np.random.lognormal(self.capacity_mu, self.capacity_sigma)
            
            # Snap to nearest tier
            tier_idx = np.argmin(np.abs(np.array(self.CAPACITY_TIERS) - base_cap))
            capacity = self.CAPACITY_TIERS[tier_idx]
            
            # Ensure minimum capacity for leaf-parent links
            if link_type in ['parent-child', 'child-parent']:
                if u_role == 'non-core' or v_role == 'non-core':
                    capacity = max(capacity, self.min_leaf_capacity)
            
            # Core links get higher capacity
            if link_type == 'core':
                capacity = max(capacity, 40.0)
                
            capacities.append(capacity)
            
        return np.array(capacities)
    
    def _calculate_prop_delays(self, edge_df: pd.DataFrame) -> np.ndarray:
        """Calculate propagation delays based on distance"""
        # Convert distance to propagation delay
        prop_delays = edge_df['dist_km'] / self.PROP_SPEED_KM_PER_MS
        
        # Add fixed processing delay
        prop_delays += self.BASE_PROCESSING_DELAY_MS
        
        return prop_delays.values
    
    def build_graphblas_matrices(self, link_table: pd.DataFrame) -> Dict:
        """
        Build GraphBLAS sparse matrices for efficient computation
        
        Returns:
            Dictionary with capacity and delay matrices
        """
        if gb is None:
            return self._build_numpy_matrices(link_table)
            
        # Get number of nodes
        n_nodes = max(link_table['u'].max(), link_table['v'].max()) + 1
        
        # Build capacity matrix
        capacity_matrix = gb.Matrix.sparse(gb.types.FP32, n_nodes, n_nodes)
        for _, row in link_table.iterrows():
            capacity_matrix[row['u'], row['v']] = row['capacity_gbps']
        
        # Build delay matrix  
        delay_matrix = gb.Matrix.sparse(gb.types.FP32, n_nodes, n_nodes)
        for _, row in link_table.iterrows():
            delay_matrix[row['u'], row['v']] = row['prop_delay_ms']
            
        return {
            'capacity': capacity_matrix,
            'delay': delay_matrix,
            'n_nodes': n_nodes
        }
    
    def _build_numpy_matrices(self, link_table: pd.DataFrame) -> Dict:
        """Fallback using NumPy sparse matrices"""
        from scipy.sparse import csr_matrix
        
        n_nodes = max(link_table['u'].max(), link_table['v'].max()) + 1
        
        # Build capacity matrix
        cap_data = link_table['capacity_gbps'].values
        cap_row = link_table['u'].values
        cap_col = link_table['v'].values
        capacity_matrix = csr_matrix((cap_data, (cap_row, cap_col)), 
                                   shape=(n_nodes, n_nodes))
        
        # Build delay matrix
        delay_data = link_table['prop_delay_ms'].values
        delay_matrix = csr_matrix((delay_data, (cap_row, cap_col)),
                                shape=(n_nodes, n_nodes))
        
        return {
            'capacity': capacity_matrix,
            'delay': delay_matrix,
            'n_nodes': n_nodes
        }
    
    @staticmethod
    def queueing_delay(utilization: float, rtt_min: float) -> float:
        """
        Calculate queueing delay using M/M/1 approximation
        
        Args:
            utilization: Link utilization (0-1)
            rtt_min: Minimum RTT for the path (ms)
            
        Returns:
            Queueing delay in ms
        """
        if utilization >= 0.99:
            return 100.0 * rtt_min  # Cap at 100x min RTT
            
        return (utilization * rtt_min) / (1 - utilization)