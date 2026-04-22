"""
PathFinder service for enumerating SCION paths - Version 2

This version properly handles SCION path structure with interface information.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import networkx as nx


@dataclass(frozen=True)
class PathHop:
    """Represents a hop in a SCION path with interfaces"""
    as_id: int
    ingress_if: Optional[int]
    egress_if: Optional[int]
    
    def __str__(self):
        if self.ingress_if is not None and self.egress_if is not None:
            return f"-[{self.ingress_if}]→AS{self.as_id}→[{self.egress_if}]-"
        elif self.ingress_if is None:
            return f"AS{self.as_id}→[{self.egress_if}]-"
        else:
            return f"-[{self.ingress_if}]→AS{self.as_id}"


@dataclass(frozen=True)  
class SCIONPath:
    """Immutable SCION path representation"""
    src: int
    dst: int
    hops: Tuple[PathHop, ...]
    path_type: str  # 'direct', 'up-down', 'up-core-down', 'peering'
    segment_ids: Tuple[str, ...]  # IDs of segments used
    
    @property
    def total_hops(self) -> int:
        return len(self.hops) - 1  # Number of links
        
    @property
    def as_sequence(self) -> Tuple[int, ...]:
        return tuple(hop.as_id for hop in self.hops)
        
    def __str__(self):
        return "".join(str(hop) for hop in self.hops)


class PathFinderV2:
    """Find paths between AS pairs using SCION segment combinations"""
    
    def __init__(self, 
                 topology_path: Path,
                 segment_store_path: Path,
                 link_table_path: Path):
        """
        Args:
            topology_path: Path to topology pickle
            segment_store_path: Path to segment store pickle
            link_table_path: Path to link table
        """
        # Load topology
        with open(topology_path, 'rb') as f:
            self.topology = pickle.load(f)
            
        self.node_df = self.topology['nodes']
        self.edge_df = self.topology['edges']
        self.n_nodes = len(self.node_df)
        
        # Create AS to ISD mapping
        self.as_to_isd = {row['as_id']: row['isd'] 
                         for _, row in self.node_df.iterrows()}
        self.core_ases = set(self.node_df[self.node_df['role'] == 'core']['as_id'])
        
        # Load segments
        with open(segment_store_path, 'rb') as f:
            self.segments = pickle.load(f)
            
        # Load link table (CLI historically used .parquet extension with pickle bytes)
        ltp = Path(link_table_path)
        if ltp.suffix.lower() == ".parquet":
            try:
                self.link_table = pd.read_parquet(link_table_path)
            except Exception:
                self.link_table = pd.read_pickle(link_table_path)
        else:
            self.link_table = pd.read_pickle(link_table_path)
        
        # Build graph for direct path checking
        self._build_graph()
        
        # Precompute edge weights for path scoring
        self._precompute_edge_weights()
        
    def _build_graph(self):
        """Build NetworkX graph for direct path checking"""
        self.G = nx.MultiDiGraph()
        
        for _, node in self.node_df.iterrows():
            self.G.add_node(node['as_id'], **node.to_dict())
            
        for _, edge in self.edge_df.iterrows():
            self.G.add_edge(edge['u'], edge['v'],
                          u_if=edge['u_if'], v_if=edge['v_if'],
                          type=edge['type'])
                          
    def _precompute_edge_weights(self):
        """Precompute edge weights for path scoring"""
        self.edge_weights = {}
        
        for _, row in self.link_table.iterrows():
            self.edge_weights[(row['u'], row['v'])] = {
                'latency': row['prop_delay_ms'],
                'capacity': row['capacity_gbps']
            }
            
    def get_paths(self, src: int, dst: int, k: int = 16, 
                  policy: str = "min-lat") -> List[SCIONPath]:
        """
        Find k best paths from src to dst
        
        Args:
            src: Source AS ID
            dst: Destination AS ID  
            k: Maximum number of paths to return
            policy: Path selection policy (min-lat, min-hop, etc.)
            
        Returns:
            List of SCIONPath objects sorted by policy
        """
        if src == dst:
            return []
            
        paths = []
        
        # 1. Check direct path
        direct_path = self._find_direct_path(src, dst)
        if direct_path:
            paths.append(direct_path)
            
        # 2. Find segment-based paths
        src_isd = self.as_to_isd[src]
        dst_isd = self.as_to_isd[dst]
        
        if src_isd == dst_isd:
            # Intra-ISD paths
            intra_paths = self._find_intra_isd_paths(src, dst, src_isd)
            paths.extend(intra_paths)
        else:
            # Inter-ISD paths
            inter_paths = self._find_inter_isd_paths(src, dst, src_isd, dst_isd)
            paths.extend(inter_paths)
            
        # 3. Score and sort paths
        scored_paths = []
        for path in paths:
            score = self._score_path(path, policy)
            scored_paths.append((score, path))
            
        scored_paths.sort(key=lambda x: x[0])
        
        # Return top k
        return [path for _, path in scored_paths[:k]]
    
    def _find_direct_path(self, src: int, dst: int) -> Optional[SCIONPath]:
        """Find direct path if it exists"""
        if not self.G.has_edge(src, dst):
            return None
            
        # Get edge data (use first edge if multiple)
        edge_data = list(self.G.get_edge_data(src, dst).values())[0]
        
        hops = [
            PathHop(src, None, edge_data['u_if']),
            PathHop(dst, edge_data['v_if'], None)
        ]
        
        return SCIONPath(
            src=src,
            dst=dst,
            hops=tuple(hops),
            path_type='direct',
            segment_ids=()
        )
    
    def _find_intra_isd_paths(self, src: int, dst: int, isd: int) -> List[SCIONPath]:
        """Find paths within a single ISD using up-down segments"""
        paths = []
        
        up_segments = self.segments['up_segments_by_isd'].get(isd, [])
        down_segments = self.segments['down_segments_by_isd'].get(isd, [])
        core_segments = self.segments['core_segments']
        
        # Find up segments from src
        src_up = [s for s in up_segments if s['src'] == src]
        
        # Find down segments to dst
        dst_down = [s for s in down_segments if s['dst'] == dst]
        
        # Try direct up-down combination
        for up_seg in src_up:
            up_core = up_seg['dst']
            
            for down_seg in dst_down:
                if down_seg['src'] == up_core:
                    # Valid up-down path
                    path = self._combine_segments(
                        [up_seg['hops'], down_seg['hops'][1:]],  # Skip duplicate core
                        'up-down',
                        (f"up_{isd}_{src}_{up_core}", f"down_{isd}_{up_core}_{dst}")
                    )
                    if path:
                        paths.append(path)
        
        # Try up-core-down within same ISD
        for up_seg in src_up:
            up_core = up_seg['dst']
            
            for core_seg in core_segments:
                if core_seg['src'] == up_core:
                    core_dst = core_seg['dst']
                    
                    for down_seg in dst_down:
                        if down_seg['src'] == core_dst:
                            # Valid up-core-down path within ISD
                            path = self._combine_segments(
                                [up_seg['hops'], core_seg['hops'][1:], down_seg['hops'][1:]],
                                'up-core-down',
                                (f"up_{isd}_{src}_{up_core}", 
                                 f"core_{up_core}_{core_dst}",
                                 f"down_{isd}_{core_dst}_{dst}")
                            )
                            if path:
                                paths.append(path)
                        
        return paths
    
    def _find_inter_isd_paths(self, src: int, dst: int, 
                             src_isd: int, dst_isd: int) -> List[SCIONPath]:
        """Find paths between different ISDs using up-core-down"""
        paths = []
        
        up_segments = self.segments['up_segments_by_isd'].get(src_isd, [])
        core_segments = self.segments['core_segments']
        down_segments = self.segments['down_segments_by_isd'].get(dst_isd, [])
        
        # Find valid combinations
        src_up = [s for s in up_segments if s['src'] == src]
        dst_down = [s for s in down_segments if s['dst'] == dst]
        
        for up_seg in src_up:
            up_core = up_seg['dst']
            
            for core_seg in core_segments:
                if core_seg['src'] == up_core:
                    core_dst = core_seg['dst']
                    
                    for down_seg in dst_down:
                        if down_seg['src'] == core_dst:
                            # Valid up-core-down path
                            path = self._combine_segments(
                                [up_seg['hops'], core_seg['hops'][1:], down_seg['hops'][1:]],
                                'up-core-down',
                                (f"up_{src_isd}_{src}_{up_core}", 
                                 f"core_{up_core}_{core_dst}",
                                 f"down_{dst_isd}_{core_dst}_{dst}")
                            )
                            if path:
                                paths.append(path)
                                
        return paths
    
    def _combine_segments(self, segment_paths: List[List[Dict]], 
                         path_type: str,
                         segment_ids: Tuple[str, ...]) -> Optional[SCIONPath]:
        """Combine segment dictionaries into a SCIONPath"""
        # Flatten hops
        all_hops = []
        for segment in segment_paths:
            for hop_dict in segment:
                hop = PathHop(
                    as_id=hop_dict['as'],
                    ingress_if=hop_dict['ingress'],
                    egress_if=hop_dict['egress']
                )
                all_hops.append(hop)
                
        if not all_hops:
            return None
            
        return SCIONPath(
            src=all_hops[0].as_id,
            dst=all_hops[-1].as_id,
            hops=tuple(all_hops),
            path_type=path_type,
            segment_ids=segment_ids
        )
    
    def _score_path(self, path: SCIONPath, policy: str) -> float:
        """Score path according to policy"""
        if policy == "min-lat":
            # Sum of edge latencies
            score = 0.0
            for i in range(len(path.hops) - 1):
                u = path.hops[i].as_id
                v = path.hops[i + 1].as_id
                if (u, v) in self.edge_weights:
                    score += self.edge_weights[(u, v)]['latency']
                else:
                    score += 100.0  # Penalty for unknown edge
            return score
            
        elif policy == "min-hop":
            return float(path.total_hops)
            
        elif policy == "max-bw":
            # Minimum capacity along path (bottleneck)
            min_bw = float('inf')
            for i in range(len(path.hops) - 1):
                u = path.hops[i].as_id
                v = path.hops[i + 1].as_id
                if (u, v) in self.edge_weights:
                    min_bw = min(min_bw, self.edge_weights[(u, v)]['capacity'])
                else:
                    min_bw = 0.0
            return -min_bw  # Negative because we want to maximize
            
        else:
            # Default to latency
            return self._score_path(path, "min-lat")
    
    def print_path_details(self, path: SCIONPath):
        """Print detailed path information"""
        print(f"Path from AS{path.src} to AS{path.dst}")
        print(f"  Type: {path.path_type}")
        print(f"  Hops: {path.total_hops}")
        print(f"  AS sequence: {' → '.join(map(str, path.as_sequence))}")
        print(f"  Full path: {path}")
        print(f"  Segments used: {path.segment_ids}")


def generate_test_segments(topology_path: Path, output_path: Path):
    """Generate a more complete set of test segments"""
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
        
    node_df = topology['nodes']
    edge_df = topology['edges']
    
    # Build graph
    G = nx.MultiDiGraph()
    for _, node in node_df.iterrows():
        G.add_node(node['as_id'], **node.to_dict())
        
    for _, edge in edge_df.iterrows():
        G.add_edge(edge['u'], edge['v'],
                  u_if=edge['u_if'], v_if=edge['v_if'],
                  type=edge['type'])
    
    # Generate segments more aggressively
    segment_store = {
        'core_segments': [],
        'up_segments_by_isd': {},
        'down_segments_by_isd': {}
    }
    
    # Core segments between all core pairs
    core_ases = list(node_df[node_df['role'] == 'core']['as_id'])
    for i, src in enumerate(core_ases):
        for dst in core_ases[i+1:]:
            if nx.has_path(G, src, dst):
                paths = list(nx.all_shortest_paths(G, src, dst))[:3]
                for path in paths:
                    segment = create_segment_from_path(G, path)
                    segment_store['core_segments'].append({
                        'path': segment,
                        'type': 'core',
                        'src': src,
                        'dst': dst
                    })
    
    # Intra-ISD segments - be more aggressive
    for isd in sorted(node_df['isd'].unique()):
        isd_nodes = node_df[node_df['isd'] == isd]
        isd_subgraph = G.subgraph(isd_nodes['as_id'].tolist())
        
        cores = list(isd_nodes[isd_nodes['role'] == 'core']['as_id'])
        non_cores = list(isd_nodes[isd_nodes['role'] == 'non-core']['as_id'])
        
        up_segs = []
        down_segs = []
        
        # Try to connect every non-core to every core
        for nc in non_cores:
            for c in cores:
                if nx.has_path(isd_subgraph, nc, c):
                    paths = list(nx.all_shortest_paths(isd_subgraph, nc, c))[:2]
                    for path in paths:
                        up_seg = create_segment_from_path(G, path)
                        up_segs.append({
                            'path': up_seg,
                            'type': 'up',
                            'src': nc,
                            'dst': c,
                            'isd': isd
                        })
                        
                        # Create down segment
                        down_seg = reverse_segment(up_seg)
                        down_segs.append({
                            'path': down_seg,
                            'type': 'down', 
                            'src': c,
                            'dst': nc,
                            'isd': isd
                        })
        
        segment_store['up_segments_by_isd'][isd] = up_segs
        segment_store['down_segments_by_isd'][isd] = down_segs
    
    with open(output_path, 'wb') as f:
        pickle.dump(segment_store, f)
        
    return segment_store


def create_segment_from_path(G, path):
    """Create segment with interface info from path"""
    segment = []
    for i in range(len(path)):
        as_id = path[i]
        
        # Get interfaces
        if i == 0:
            ingress_if = None
        else:
            edge_data = G.get_edge_data(path[i-1], as_id)
            if edge_data:
                first_edge = list(edge_data.values())[0]
                ingress_if = first_edge['v_if']
            else:
                ingress_if = None
        
        if i == len(path) - 1:
            egress_if = None
        else:
            edge_data = G.get_edge_data(as_id, path[i+1])
            if edge_data:
                first_edge = list(edge_data.values())[0]
                egress_if = first_edge['u_if']
            else:
                egress_if = None
        
        segment.append({
            'as': as_id,
            'ingress': ingress_if,
            'egress': egress_if
        })
    
    return segment


def reverse_segment(segment):
    """Reverse a segment, swapping ingress/egress"""
    reversed_seg = []
    for hop in reversed(segment):
        reversed_seg.append({
            'as': hop['as'],
            'ingress': hop['egress'],
            'egress': hop['ingress']
        })
    return reversed_seg