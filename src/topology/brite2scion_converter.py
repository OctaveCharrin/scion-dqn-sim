"""
BRITE to SCION topology converter

Converts BRITE topologies to SCION format with:
- ISD assignment using k-means clustering
- Core AS selection based on degree centrality
- Link type classification
- Interface ID management
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.cluster import KMeans
import pickle


class BRITE2SCIONConverter:
    """Convert BRITE topologies to SCION format"""
    
    def __init__(self, 
                 n_isds: int = 3,
                 core_ratio: float = 0.075):
        """
        Args:
            n_isds: Number of ISDs to create (automatically reduced to 1 for topologies < 200 ASes)
            core_ratio: Fraction of ASes to designate as core per ISD (default 7.5%, bounded to 5-10%)
        """
        self.n_isds = n_isds
        self.original_n_isds = n_isds  # Store original value
        # Bound core_ratio to 5-10%
        self.core_ratio = max(0.05, min(0.10, core_ratio))
        
    def convert(self, brite_file: Path, output_path: Path) -> Dict:
        """
        Convert BRITE topology to SCION format
        
        Args:
            brite_file: Path to BRITE .edges file
            output_path: Where to save the converted topology
            
        Returns:
            Dictionary with topology data
        """
        # Read BRITE topology
        G, node_attrs = self._read_brite_edges(brite_file)
        for n in G.nodes():
            G.nodes[n]["x"] = float(node_attrs[n]["x"])
            G.nodes[n]["y"] = float(node_attrs[n]["y"])

        # Automatically adjust n_isds for small topologies
        topology_size = len(G.nodes())
        if topology_size < 200:
            # For small topologies, use only 1 ISD
            self.n_isds = 1
            print(f"Topology has {topology_size} ASes (< 200), using 1 ISD")
        else:
            # Keep original n_isds for larger topologies
            print(f"Topology has {topology_size} ASes (>= 200), using {self.n_isds} ISDs")
        
        # Assign ISDs using geographic k-means
        isd_assignment = self._assign_isds(G, node_attrs)
        
        # Select core ASes per ISD
        core_ases = self._select_core_ases(G, isd_assignment)
        
        # Ensure core connectivity
        self._ensure_core_connectivity(G, core_ases)
        
        # Ensure multi-parent connectivity for path diversity
        # Note: This should be called BEFORE the automatic multi-core connectivity
        # in _ensure_core_connectivity to be more effective
        self._ensure_multi_parent_connectivity(G, core_ases, isd_assignment)
        
        # Add additional cross-connections for dense topology
        self._add_dense_connections(G, core_ases, isd_assignment)
        
        # Classify links
        link_types = self._classify_links(G, core_ases, isd_assignment)
        
        # Assign interface IDs
        interface_ids = self._assign_interface_ids(G)
        
        # Build node and edge DataFrames
        node_df = self._build_node_dataframe(G, node_attrs, isd_assignment, core_ases)
        edge_df = self._build_edge_dataframe(G, link_types, interface_ids)
        
        # Calculate distances for edges
        edge_df = self._add_edge_distances(edge_df, node_df)
        
        # Save topology
        topology = {
            'nodes': node_df,
            'edges': edge_df,
            'metadata': {
                'n_nodes': len(node_df),
                'n_edges': len(edge_df),
                'n_isds': self.n_isds,
                'n_core_ases': len(core_ases)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(topology, f)
        
        # Restore original n_isds value to avoid side effects
        self.n_isds = self.original_n_isds
            
        return topology

    def convert_brite_file(self, brite_file: Path) -> Dict:
        """
        Load a BRITE ``.brite`` export, run SCION assignment/classification, and return
        the structure expected by ``evaluation/01_generate_topology.py`` (graph + isds + core_ases).
        """
        brite_file = Path(brite_file)
        G, node_attrs = self._read_brite_edges(brite_file)
        for n in G.nodes():
            G.nodes[n]["x"] = float(node_attrs[n]["x"])
            G.nodes[n]["y"] = float(node_attrs[n]["y"])

        topology_size = len(G.nodes())
        if topology_size < 200:
            self.n_isds = 1
            print(f"Topology has {topology_size} ASes (< 200), using 1 ISD")
        else:
            print(f"Topology has {topology_size} ASes (>= 200), using {self.n_isds} ISDs")

        isd_assignment = self._assign_isds(G, node_attrs)
        core_ases = self._select_core_ases(G, isd_assignment)
        self._ensure_core_connectivity(G, core_ases)
        self._ensure_multi_parent_connectivity(G, core_ases, isd_assignment)
        self._add_dense_connections(G, core_ases, isd_assignment)
        link_types = self._classify_links(G, core_ases, isd_assignment)

        for n in G.nodes():
            G.nodes[n]["isd"] = int(isd_assignment[n])

        for u, v in G.edges():
            lt = link_types.get((u, v)) or link_types.get((v, u))
            if lt is None:
                lt = "peer"
            delay = float(G[u][v].get("delay", 1.0))
            bw = float(G[u][v].get("bandwidth", 10.0))
            G[u][v]["type"] = lt.upper().replace("-", "_")
            G[u][v]["latency"] = delay
            G[u][v]["bandwidth"] = bw

        isds = [
            {
                "isd_id": int(isd_id),
                "member_ases": sorted(
                    n for n, k in isd_assignment.items() if k == isd_id
                ),
            }
            for isd_id in sorted(set(isd_assignment.values()))
        ]

        self.n_isds = self.original_n_isds

        return {
            "graph": G,
            "isds": isds,
            "core_ases": set(core_ases),
        }
    
    def _read_brite_edges(self, brite_file: Path) -> Tuple[nx.Graph, Dict]:
        """Read BRITE topology file (BriteExport format) and extract node positions and edges."""
        G = nx.Graph()
        node_attrs: Dict[int, Dict[str, float]] = {}
        
        with open(brite_file) as f:
            in_nodes = False
            in_edges = False
            
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lower = line.lower()
                if lower.startswith("topology:"):
                    continue
                if lower.startswith("nodes:"):
                    in_nodes = True
                    in_edges = False
                    continue
                if lower.startswith("edges:"):
                    in_nodes = False
                    in_edges = True
                    continue
                    
                if in_nodes:
                    parts = line.split("\t") if "\t" in line else line.split()
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        node_attrs[node_id] = {"x": x, "y": y}
                        G.add_node(node_id)
                        
                elif in_edges:
                    parts = line.split("\t") if "\t" in line else line.split()
                    if len(parts) >= 6:
                        u = int(parts[1])
                        v = int(parts[2])
                        dist = float(parts[3])
                        delay = float(parts[4])
                        bw = float(parts[5])
                        G.add_edge(u, v, bandwidth=bw, delay=delay, length=dist)
        
        return G, node_attrs
    
    def _assign_isds(self, G: nx.Graph, node_attrs: Dict) -> Dict[int, int]:
        """Assign nodes to ISDs using k-means on geographic coordinates"""
        nodes = sorted(G.nodes())
        
        # Special case: if only 1 ISD, assign all nodes to ISD 0
        if self.n_isds == 1:
            return {node: 0 for node in nodes}
        
        # Otherwise, use k-means clustering
        coords = np.array([[node_attrs[n]['x'], node_attrs[n]['y']] for n in nodes])
        
        kmeans = KMeans(n_clusters=self.n_isds, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        return {node: int(label) for node, label in zip(nodes, labels)}
    
    def _select_core_ases(self, G: nx.Graph, isd_assignment: Dict) -> set:
        """Select core ASes based on degree centrality per ISD (5-10% of ASes per ISD)"""
        core_ases = set()
        
        # Group nodes by ISD
        isd_nodes = {}
        for node, isd in isd_assignment.items():
            if isd not in isd_nodes:
                isd_nodes[isd] = []
            isd_nodes[isd].append(node)
        
        # Select top degree nodes per ISD
        for isd, nodes in isd_nodes.items():
            subgraph = G.subgraph(nodes)
            degrees = dict(subgraph.degree())
            
            # Calculate number of core ASes (5-10% of ASes in this ISD)
            n_core = max(1, int(len(nodes) * self.core_ratio))
            
            # Ensure we don't exceed 10% even with rounding
            max_core = max(1, int(len(nodes) * 0.10))
            n_core = min(n_core, max_core)
            
            # Sort by degree and select top nodes
            sorted_nodes = sorted(nodes, key=lambda n: degrees[n], reverse=True)
            selected_core = sorted_nodes[:n_core]
            core_ases.update(selected_core)
            
            print(f"  ISD {isd}: {len(nodes)} ASes, {n_core} core ASes ({n_core/len(nodes)*100:.1f}%)")
            
        return core_ases
    
    def _ensure_core_connectivity(self, G: nx.Graph, core_ases: set):
        """Ensure core ASes form a connected subgraph by adding virtual links if needed"""
        core_list = list(core_ases)
        n_cores = len(core_list)
        
        if n_cores <= 1:
            return
            
        # Check existing core connectivity
        core_subgraph = G.subgraph(core_list)
        
        # Ensure full mesh connectivity for small number of cores
        if n_cores <= 5:
            print("  Ensuring full mesh core connectivity...")
            added = 0
            for i in range(n_cores):
                for j in range(i + 1, n_cores):
                    u = core_list[i]
                    v = core_list[j]
                    
                    if not G.has_edge(u, v):
                        # Add virtual core link with high bandwidth and low latency
                        G.add_edge(u, v, bandwidth=100000.0, virtual=True)
                        print(f"    Added virtual core link: {u} <-> {v}")
                        added += 1
            
            if added == 0:
                print("    Core ASes already fully connected")
        
        # Skip the aggressive multi-core connectivity to preserve hierarchy
        # Path diversity will come from multi-parent connections instead
        
        # For larger number of cores, ensure basic connectivity
        if n_cores > 5:
            if not nx.is_connected(core_subgraph):
                print("  Adding virtual core links to ensure connectivity...")
                # Create a ring topology for core ASes
                for i in range(n_cores):
                    u = core_list[i]
                    v = core_list[(i + 1) % n_cores]
                    
                    if not G.has_edge(u, v):
                        # Add virtual core link with high bandwidth and low latency
                        G.add_edge(u, v, bandwidth=100000.0, virtual=True)
                        print(f"    Added virtual core link: {u} <-> {v}")
    
    def _ensure_multi_parent_connectivity(self, G: nx.Graph, core_ases: set, 
                                        isd_assignment: Dict):
        """Ensure more ASes have multiple parents for better path diversity"""
        print("  Ensuring multi-parent connectivity for path diversity...")
        
        # Group nodes by ISD
        isd_nodes = {}
        for node, isd in isd_assignment.items():
            if isd not in isd_nodes:
                isd_nodes[isd] = []
            isd_nodes[isd].append(node)
        
        added_connections = 0
        
        for isd, nodes in isd_nodes.items():
            # Get core and non-core ASes in this ISD
            isd_cores = [n for n in nodes if n in core_ases]
            non_cores = [n for n in nodes if n not in core_ases]
            
            if not isd_cores or not non_cores:
                continue
            
            # For each non-core AS, check parent connectivity
            for node in non_cores:
                # Find current distance to nearest core
                dist_to_core = self._distance_to_core(node, G, core_ases, isd)
                
                if dist_to_core > 2:  # Far from core, needs better connectivity
                    # Find potential parents (ASes closer to core)
                    potential_parents = []
                    
                    for other in nodes:
                        if other == node or other in core_ases:
                            continue
                            
                        other_dist = self._distance_to_core(other, G, core_ases, isd)
                        if other_dist < dist_to_core - 1:  # Significantly closer to core
                            # Check if not already connected
                            if not G.has_edge(node, other):
                                potential_parents.append((other, other_dist))
                    
                    # Sort by distance to core
                    potential_parents.sort(key=lambda x: x[1])
                    
                    # Add connections to 1-2 closest potential parents
                    for i, (parent, _) in enumerate(potential_parents[:2]):
                        # Calculate geographic distance
                        node_pos = (G.nodes[node].get('x', 0), G.nodes[node].get('y', 0))
                        parent_pos = (G.nodes[parent].get('x', 0), G.nodes[parent].get('y', 0))
                        geo_dist = np.sqrt((node_pos[0] - parent_pos[0])**2 + 
                                         (node_pos[1] - parent_pos[1])**2)
                        
                        # Only add if geographically reasonable
                        if geo_dist < 400:  # Reasonable distance threshold
                            G.add_edge(node, parent, bandwidth=1000.0, virtual=True)
                            added_connections += 1
                            print(f"    Added multi-parent link: {node} -> {parent}")
                
                # Also ensure ASes at distance 1 and 2 have multiple paths
                elif dist_to_core in [1, 2]:
                    # Count current parents (including direct core connections)
                    parents = []
                    for neighbor in G.neighbors(node):
                        if neighbor in core_ases:
                            parents.append(neighbor)
                        else:
                            neighbor_dist = self._distance_to_core(neighbor, G, core_ases, isd)
                            if neighbor_dist < dist_to_core:
                                parents.append(neighbor)
                    
                    # Add more parents - target based on topology size
                    if len(nodes) < 30:
                        target_parents = 2  # For small topologies
                    else:
                        target_parents = 3 if dist_to_core == 1 else 2
                    if len(parents) < target_parents:
                        # Find other ASes at same or closer distance
                        candidates = []
                        
                        for other in nodes:
                            if other == node or other in parents:
                                continue
                            
                            # Include cores and ASes closer to core
                            if other in core_ases:
                                other_dist = 0
                            else:
                                other_dist = self._distance_to_core(other, G, core_ases, isd)
                            
                            if other_dist < dist_to_core and not G.has_edge(node, other):
                                # Check geographic distance
                                node_pos = (G.nodes[node].get('x', 0), G.nodes[node].get('y', 0))
                                other_pos = (G.nodes[other].get('x', 0), G.nodes[other].get('y', 0))
                                geo_dist = np.sqrt((node_pos[0] - other_pos[0])**2 + 
                                                 (node_pos[1] - other_pos[1])**2)
                                
                                candidates.append((other, other_dist, geo_dist))
                        
                        # Sort by distance to core, then geographic distance
                        candidates.sort(key=lambda x: (x[1], x[2]))
                        
                        # Add connections to multiple candidates
                        parents_to_add = min(target_parents - len(parents), len(candidates))
                        for i in range(parents_to_add):
                            if candidates[i][2] < 500:  # Increased geographic distance threshold
                                best = candidates[i][0]
                                G.add_edge(node, best, bandwidth=2000.0, virtual=True)
                                added_connections += 1
                                print(f"    Added parent {len(parents) + i + 1}: {node} -> {best}")
        
        print(f"    Total multi-parent connections added: {added_connections}")
    
    def _add_dense_connections(self, G: nx.Graph, core_ases: set,
                               isd_assignment: Dict) -> None:
        """Add additional cross-connections for dense topology with many paths"""
        print("  Adding dense cross-connections...")
        added_connections = 0
        
        # Group ASes by ISD and distance from core
        as_by_distance = {}
        for isd_id in set(isd_assignment.values()):
            as_by_distance[isd_id] = {}
            
            for as_id, isd in isd_assignment.items():
                if isd != isd_id:
                    continue
                    
                dist = self._distance_to_core(as_id, G, core_ases, isd_id)
                if dist not in as_by_distance[isd_id]:
                    as_by_distance[isd_id][dist] = []
                as_by_distance[isd_id][dist].append(as_id)
        
        # Add cross-connections between ASes at the same level
        for isd_id, distance_groups in as_by_distance.items():
            for dist, ases in distance_groups.items():
                if dist == 0:  # Skip core ASes
                    continue
                    
                # Connect ASes at same distance with some probability
                for i, as1 in enumerate(ases):
                    for j, as2 in enumerate(ases[i+1:], i+1):
                        if not G.has_edge(as1, as2):
                            # Check geographic distance
                            pos1 = (G.nodes[as1].get('x', 0), G.nodes[as1].get('y', 0))
                            pos2 = (G.nodes[as2].get('x', 0), G.nodes[as2].get('y', 0))
                            geo_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                            
                            # Add connection with probability based on distance
                            if geo_dist < 200 and np.random.random() < 0.15:
                                G.add_edge(as1, as2, bandwidth=1000.0, virtual=True, cross_connect=True)
                                added_connections += 1
        
        # Add shortcut connections between different levels
        for isd_id, distance_groups in as_by_distance.items():
            levels = sorted(distance_groups.keys())
            for i, level1 in enumerate(levels[:-1]):
                if level1 == 0:  # Skip core level
                    continue
                    
                for level2 in levels[i+1:]:
                    if level2 - level1 > 2:  # Skip if too far apart
                        continue
                        
                    # Connect some ASes between levels
                    ases1 = distance_groups[level1]
                    ases2 = distance_groups[level2]
                    
                    for as1 in ases1[:5]:  # Limit connections
                        for as2 in ases2[:5]:
                            if not G.has_edge(as1, as2):
                                pos1 = (G.nodes[as1].get('x', 0), G.nodes[as1].get('y', 0))
                                pos2 = (G.nodes[as2].get('x', 0), G.nodes[as2].get('y', 0))
                                geo_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                                
                                if geo_dist < 250 and np.random.random() < 0.1:
                                    G.add_edge(as1, as2, bandwidth=1000.0, virtual=True, shortcut=True)
                                    added_connections += 1
        
        print(f"    Total dense connections added: {added_connections}")
    
    def _classify_links(self, G: nx.Graph, core_ases: set, 
                       isd_assignment: Dict) -> Dict[Tuple[int, int], str]:
        """Classify links as core or parent-child (no peering links)"""
        link_types = {}
        
        # First, ensure all non-core ASes have a path to core
        # Build a tree structure for each ISD
        isd_trees = self._build_isd_trees(G, core_ases, isd_assignment)
        
        for u, v in G.edges():
            u_core = u in core_ases
            v_core = v in core_ases
            u_isd = isd_assignment[u]
            v_isd = isd_assignment[v]
            
            if u_core and v_core:
                # Both core -> core link
                link_types[(u, v)] = 'core'
            elif u_core and not v_core and u_isd == v_isd:
                # Core to non-core in same ISD -> parent-child
                link_types[(u, v)] = 'parent-child'
            elif not u_core and v_core and u_isd == v_isd:
                # Non-core to core in same ISD -> child-parent
                link_types[(u, v)] = 'child-parent'
            elif not u_core and not v_core and u_isd == v_isd:
                # Determine parent-child direction based on distance to core
                u_dist = self._distance_to_core(u, G, core_ases, u_isd)
                v_dist = self._distance_to_core(v, G, core_ases, v_isd)
                
                if u_dist < v_dist:
                    link_types[(u, v)] = 'parent-child'
                elif v_dist < u_dist:
                    link_types[(u, v)] = 'child-parent'
                else:
                    # Same level - make it parent-child based on AS ID for consistency
                    if u < v:
                        link_types[(u, v)] = 'parent-child'
                    else:
                        link_types[(u, v)] = 'child-parent'
            # Cross-ISD non-core links are not allowed in SCION
                
        return link_types
    
    def _build_isd_trees(self, G: nx.Graph, core_ases: set, 
                        isd_assignment: Dict) -> Dict[int, set]:
        """Build minimum spanning trees to ensure connectivity to core ASes"""
        isd_trees = {}
        
        # Group nodes by ISD
        isd_nodes = {}
        for node, isd in isd_assignment.items():
            if isd not in isd_nodes:
                isd_nodes[isd] = []
            isd_nodes[isd].append(node)
        
        # For each ISD, build a tree connecting all non-core to core
        for isd, nodes in isd_nodes.items():
            isd_core = [n for n in nodes if n in core_ases]
            isd_non_core = [n for n in nodes if n not in core_ases]
            
            if not isd_core or not isd_non_core:
                isd_trees[isd] = set()
                continue
            
            # Use BFS to build tree from core ASes
            tree_edges = set()
            visited = set(isd_core)
            queue = list(isd_core)
            
            while queue and len(visited) < len(nodes):
                current = queue.pop(0)
                
                # Check all neighbors in the same ISD
                for neighbor in G.neighbors(current):
                    if neighbor not in visited and neighbor in nodes:
                        visited.add(neighbor)
                        tree_edges.add((current, neighbor))
                        queue.append(neighbor)
            
            isd_trees[isd] = tree_edges
            
        return isd_trees
    
    def _distance_to_core(self, node: int, G: nx.Graph, 
                         core_ases: set, isd: int) -> int:
        """Calculate shortest path distance to nearest core AS in same ISD"""
        if node in core_ases:
            return 0
            
        # BFS to find shortest path to any core AS
        visited = {node}
        queue = [(node, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            
            for neighbor in G.neighbors(current):
                if neighbor in visited:
                    continue
                    
                if neighbor in core_ases:
                    return dist + 1
                    
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
        
        # If no path found, return large number
        return 999
    
    def _assign_interface_ids(self, G: nx.Graph) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Assign interface IDs monotonically per AS"""
        interface_ids = {}
        as_if_counters = {}
        
        for u, v in G.edges():
            # Assign interface ID for u
            if u not in as_if_counters:
                as_if_counters[u] = 0
            u_if = as_if_counters[u]
            as_if_counters[u] += 1
            
            # Assign interface ID for v
            if v not in as_if_counters:
                as_if_counters[v] = 0
            v_if = as_if_counters[v]
            as_if_counters[v] += 1
            
            interface_ids[(u, v)] = (u_if, v_if)
            
        return interface_ids
    
    def _build_node_dataframe(self, G: nx.Graph, node_attrs: Dict,
                             isd_assignment: Dict, core_ases: set) -> pd.DataFrame:
        """Build node DataFrame with attributes"""
        data = []
        for node in G.nodes():
            data.append({
                'as_id': node,
                'isd': isd_assignment[node],
                'role': 'core' if node in core_ases else 'non-core',
                'x': node_attrs[node]['x'],
                'y': node_attrs[node]['y'],
                'degree': G.degree(node)
            })
        
        return pd.DataFrame(data)
    
    def _build_edge_dataframe(self, G: nx.Graph, link_types: Dict,
                             interface_ids: Dict) -> pd.DataFrame:
        """Build edge DataFrame with attributes"""
        data = []
        for (u, v), link_type in link_types.items():
            u_if, v_if = interface_ids[(u, v)]
            data.append({
                'u': u,
                'v': v,
                'u_if': u_if,
                'v_if': v_if,
                'type': link_type,
                'bandwidth': G[u][v].get('bandwidth', 10.0)
            })
            
        return pd.DataFrame(data)
    
    def _add_edge_distances(self, edge_df: pd.DataFrame, 
                           node_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate geographic distances for edges"""
        # Create node position lookup
        pos = {row['as_id']: (row['x'], row['y']) 
               for _, row in node_df.iterrows()}
        
        distances = []
        for _, edge in edge_df.iterrows():
            u_pos = pos[edge['u']]
            v_pos = pos[edge['v']]
            dist = np.sqrt((u_pos[0] - v_pos[0])**2 + (u_pos[1] - v_pos[1])**2)
            distances.append(dist)
            
        edge_df['dist_km'] = distances
        return edge_df