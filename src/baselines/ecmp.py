"""
Equal-Cost Multi-Path (ECMP) selection algorithm
"""

import numpy as np
from typing import List, Dict, Any

class ECMPSelector:
    """ECMP path selection with statistical load balancing"""
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select among equal-cost paths using random distribution
        to simulate 5-tuple flow hashing over time.
        """
        if not paths:
            return 0
            
        # 1. Find shortest paths (equal cost)
        hop_counts = [len(p.as_sequence) for p in paths]
        min_hops = min(hop_counts)
        shortest_indices = [i for i, h in enumerate(hop_counts) if h == min_hops]
        
        # 2. Hash the flow to a stable path among equal-cost options (5-tuple ECMP).
        flow_key = (
            flow.get("source_as"),
            flow.get("destination_as"),
            flow.get("flow_id"),
        )
        pick = hash(flow_key) % len(shortest_indices)
        return shortest_indices[pick]