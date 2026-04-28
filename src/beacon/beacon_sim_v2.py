"""
SCION beacon simulator (control plane).

Implements the two-phase SCION beaconing model:

1. **Core beaconing** — every core AS originates Path Construction Beacons
   (PCBs) and propagates them across CORE links (intra-ISD and inter-ISD).
   Generated segments are registered as ``core_segments``.
2. **Intra-ISD beaconing** — each core originates PCBs that propagate
   strictly **top-down** along ``parent-child`` links toward non-core ASes.
   The reverse direction (``child-parent``) is never traversed during
   beaconing — that direction is consumed only when the resulting
   segments are reused as up-segments later.

Peer links are never used to propagate PCBs: the path-builder consumes
them as shortcuts when assembling end-to-end paths from up/core/down
segments. To keep PCB fan-out tractable on dense topologies, both phases
support a per-originator fan-out cap (``max_segments_per_origin``) and a
BFS-pop budget (``max_intra_queue_pops``).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time


@dataclass
class PCB:
    """Path Segment Construction Beacon"""
    originator: int
    path: List[Dict[str, Any]] = field(default_factory=list)
    segment_id: str = ""
    
    def copy(self):
        """Create a copy for propagation"""
        return PCB(
            originator=self.originator,
            path=[h.copy() for h in self.path],
            segment_id=self.segment_id
        )


class CorrectedBeaconSimulator:
    """SCION beacon simulator with proper interface tracking and SCION-compliant
    propagation rules (top-down intra-ISD, peer-link skip).
    """

    def __init__(
        self,
        max_segments_per_type: int = 50,
        max_intra_queue_pops: Optional[int] = None,
        max_intra_segments_per_isd: Optional[int] = None,
        max_segments_per_origin: Optional[int] = None,
    ):
        """Initialize beacon simulator.

        Args:
            max_segments_per_type: Max segments to keep per (src, dst, type) tuple.
            max_intra_queue_pops: Cap BFS-style intra-ISD propagation work per
                originating core (avoids exponential blow-ups on dense graphs).
                Defaults to 80_000 or ``BEACON_INTRA_MAX_POPS`` if set.
            max_intra_segments_per_isd: Max registered *down* intra segments per
                ISD (each adds a matching up segment). Defaults to 20_000 or
                ``BEACON_MAX_INTRA_SEGMENTS_PER_ISD`` if set.
            max_segments_per_origin: Soft fan-out cap per originating core for
                each phase (core / intra). Mirrors the way real SCION beacon
                services keep only the top-K candidate beacons per origin.
                Defaults to 200 or ``BEACON_MAX_SEGMENTS_PER_ORIGIN`` if set.
        """
        self.max_segments_per_type = max_segments_per_type
        if max_intra_queue_pops is not None:
            self.max_intra_queue_pops = int(max_intra_queue_pops)
        else:
            env_cap = os.environ.get("BEACON_INTRA_MAX_POPS")
            self.max_intra_queue_pops = int(env_cap) if env_cap else 80_000
        if max_intra_segments_per_isd is not None:
            self.max_intra_segments_per_isd = int(max_intra_segments_per_isd)
        else:
            env_ms = os.environ.get("BEACON_MAX_INTRA_SEGMENTS_PER_ISD")
            self.max_intra_segments_per_isd = int(env_ms) if env_ms else 20_000
        if max_segments_per_origin is not None:
            self.max_segments_per_origin = int(max_segments_per_origin)
        else:
            env_so = os.environ.get("BEACON_MAX_SEGMENTS_PER_ORIGIN")
            self.max_segments_per_origin = int(env_so) if env_so else 200
        
    def simulate(self, topology_path: Path, output_dir: Path) -> Dict:
        """Run beacon simulation with proper interface tracking.

        Args:
            topology_path: Path to topology pickle.
            output_dir: Directory for outputs.

        Returns:
            ``(segment_store, stats)`` tuple where ``segment_store`` is the
            legacy dict shape consumed by the path builder.
        """
        print("=== SCION Beacon Simulation ===\n")
        self._intra_budget_warned = False
        self._intra_seg_cap_warned = False
        self._intra_segment_cap_hit = False

        # Load topology
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)

        self.node_df = topology['nodes']
        self.edge_df = topology['edges']

        # Build graph with interface info
        self.G = self._build_graph()

        # Initialize segment store
        self.segments = {
            'core': [],
            'up': defaultdict(list),    # by ISD
            'down': defaultdict(list)   # by ISD
        }
        self._core_segment_keys: set = set()
        self._intra_down_keys_by_isd: Dict[int, set] = defaultdict(set)

        # Run simulation phases
        print("Phase 1: Core Beaconing")
        core_stats = self._simulate_core_beaconing()

        print("\nPhase 2: Intra-ISD Beaconing")
        intra_stats = self._simulate_intra_beaconing()
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to legacy format for compatibility
        segment_store = self._convert_to_legacy_format()
        
        with open(output_dir / 'segments_corrected.pkl', 'wb') as f:
            pickle.dump(segment_store, f)
            
        # Save detailed segments with metadata
        with open(output_dir / 'segments_detailed.pkl', 'wb') as f:
            pickle.dump(self.segments, f)
            
        # Generate statistics
        stats = {
            'core': core_stats,
            'intra': intra_stats,
            'totals': {
                'core_segments': len(self.segments['core']),
                'up_segments': sum(len(segs) for segs in self.segments['up'].values()),
                'down_segments': sum(len(segs) for segs in self.segments['down'].values())
            }
        }
        
        with open(output_dir / 'beacon_stats_corrected.pkl', 'wb') as f:
            pickle.dump(stats, f)
            
        return segment_store, stats
    
    def _build_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph with full edge information"""
        G = nx.MultiDiGraph()
        
        # Add nodes with attributes
        for _, node in self.node_df.iterrows():
            G.add_node(node['as_id'], **node.to_dict())
            
        # Add edges with interface info
        for _, edge in self.edge_df.iterrows():
            G.add_edge(edge['u'], edge['v'],
                      u_if=edge['u_if'], 
                      v_if=edge['v_if'],
                      type=edge['type'],
                      latency=edge.get('latency', 10),
                      capacity=edge.get('capacity', 1000))
                      
        return G
    
    def _simulate_core_beaconing(self) -> Dict:
        """Simulate core AS beaconing across CORE links (intra- and inter-ISD)."""
        core_ases = list(self.node_df[self.node_df['role'] == 'core']['as_id'])
        print(f"  Core ASes: {core_ases}")

        stats = {
            'beacons_originated': 0,
            'segments_discovered': 0,
            'propagation_depths': []
        }

        for originator in core_ases:
            pcb = PCB(originator=originator)
            pcb.segment_id = f"core_{originator}_{int(time.time())}"
            pcb.path.append({
                'as': originator,
                'ingress': None,
                'egress': None,
            })
            self._propagate_core_pcb(pcb, originator, stats)
            stats['beacons_originated'] += 1

        print(f"  Originated: {stats['beacons_originated']} beacons")
        print(f"  Discovered: {stats['segments_discovered']} core segments")

        return stats
    
    def _propagate_core_pcb(self, pcb: PCB, current_as: int, stats: Dict):
        """Propagate PCB through the core network across CORE links only.

        The adapter writes both directions of each undirected edge, so we only
        traverse outgoing edges (``successors``). The previous implementation
        also iterated ``predecessors`` and a fallback ``forward=False`` branch,
        which double-counted every edge.

        Per-origin fan-out is bounded by ``self.max_segments_per_origin`` so
        dense topologies cannot produce an unbounded combinatorial expansion of
        core segments.
        """
        visited_paths = set()
        queue = deque([(current_as, pcb)])
        reached_cores = {pcb.originator}
        segments_for_origin = 0
        cap_per_origin = self.max_segments_per_origin

        while queue:
            if segments_for_origin >= cap_per_origin:
                break
            current, current_pcb = queue.popleft()
            path_sig = tuple(h['as'] for h in current_pcb.path)

            for neighbor in self.G.successors(current):
                if neighbor in path_sig:
                    continue

                # Beacons only ride CORE links; PEER edges are not used as
                # beacon carriers in the SCION control plane.
                edge_data = self.G.get_edge_data(current, neighbor)
                if not edge_data:
                    continue
                core_edge = None
                for _, edge_info in edge_data.items():
                    if edge_info.get('type') == 'core':
                        core_edge = dict(edge_info)
                        break
                if core_edge is None:
                    continue

                if self.G.nodes[neighbor].get('role') != 'core':
                    continue

                new_path_sig = path_sig + (neighbor,)
                if new_path_sig in visited_paths:
                    continue
                visited_paths.add(new_path_sig)

                new_pcb = current_pcb.copy()
                # Forward direction: u_if is the egress at ``current``,
                # v_if is the ingress at ``neighbor``.
                new_pcb.path[-1]['egress'] = core_edge.get('u_if')
                new_pcb.path.append({
                    'as': neighbor,
                    'ingress': core_edge.get('v_if'),
                    'egress': None,
                })

                if self._register_core_segment(new_pcb, stats):
                    segments_for_origin += 1
                reached_cores.add(neighbor)
                queue.append((neighbor, new_pcb))

                if segments_for_origin >= cap_per_origin:
                    break

        stats['propagation_depths'].append(len(reached_cores) - 1)
    
    def _register_core_segment(self, pcb: PCB, stats: Dict) -> bool:
        """Register a core segment. Returns True if a new segment was added."""
        segment = {
            'type': 'core',
            'src': pcb.originator,
            'dst': pcb.path[-1]['as'],
            'hops': [h.copy() for h in pcb.path],
            'path': [h['as'] for h in pcb.path],
            'segment_id': pcb.segment_id,
            'timestamp': time.time(),
        }
        key = (segment['src'], segment['dst'], tuple(segment['path']))
        if key in self._core_segment_keys:
            return False
        self._core_segment_keys.add(key)

        self.segments['core'].append(segment)
        stats['segments_discovered'] += 1
        return True
    
    def _simulate_intra_beaconing(self) -> Dict:
        """Simulate intra-ISD beaconing (strictly top-down parent→child)."""
        stats = defaultdict(lambda: {
            'beacons_originated': 0,
            'up_segments': 0,
            'down_segments': 0,
            'coverage': 0.0,
        })

        for isd in sorted(self.node_df['isd'].unique()):
            print(f"\n  ISD {isd}:")
            isd_nodes = self.node_df[self.node_df['isd'] == isd]
            isd_cores = list(isd_nodes[isd_nodes['role'] == 'core']['as_id'])
            isd_non_cores = list(isd_nodes[isd_nodes['role'] == 'non-core']['as_id'])

            if not isd_cores:
                print("    No core ASes - skipping")
                continue

            print(f"    Core ASes: {isd_cores}")
            print(f"    Non-core ASes: {len(isd_non_cores)}")
            self._intra_segment_cap_hit = False

            reachable_non_cores: set = set()

            for core_originator in isd_cores:
                if self._intra_segment_cap_hit:
                    break
                pcb = PCB(originator=core_originator)
                pcb.segment_id = (
                    f"intra_{isd}_{core_originator}_{int(time.time())}"
                )
                pcb.path.append({
                    'as': core_originator,
                    'ingress': None,
                    'egress': None,
                })
                reached = self._propagate_intra_pcb(
                    pcb, core_originator, isd, stats[isd]
                )
                reachable_non_cores.update(reached)
                stats[isd]['beacons_originated'] += 1

            if isd_non_cores:
                stats[isd]['coverage'] = (
                    len(reachable_non_cores) / len(isd_non_cores)
                )

            print(f"    Beacons originated: {stats[isd]['beacons_originated']}")
            print(f"    Up segments: {stats[isd]['up_segments']}")
            print(f"    Down segments: {stats[isd]['down_segments']}")
            print(f"    Non-core coverage: {stats[isd]['coverage']:.1%}")

        return dict(stats)
    
    def _propagate_intra_pcb(
        self, pcb: PCB, current_as: int, isd: int, stats: Dict
    ) -> set:
        """Propagate PCB within ISD strictly top-down (parent→child).

        SCION intra-ISD beaconing uses only the ``parent-child`` orientation of
        a hierarchy edge: a core sends down to its direct children, which
        re-broadcast to their own children. The ``child-parent`` direction (the
        reverse half-edge) is **never** traversed during beaconing — it only
        exists in the segment store via the symmetrically-derived up segment.

        Peer links and core links are skipped here. The simulator keeps a
        per-origin segment cap (``max_segments_per_origin``) so dense ISDs
        cannot generate an unbounded number of segments per core.
        """
        visited_paths = set()
        queue = deque([(current_as, pcb, 0)])
        reached_non_cores: set = set()
        max_depth = 10
        pops = 0
        segments_for_origin = 0
        cap_per_origin = self.max_segments_per_origin

        while queue:
            if self._intra_segment_cap_hit:
                break
            if pops >= self.max_intra_queue_pops:
                if not self._intra_budget_warned:
                    self._intra_budget_warned = True
                    print(
                        "    Warning: intra-ISD propagation budget exhausted "
                        f"({self.max_intra_queue_pops} queue pops); stopping early."
                    )
                break
            if segments_for_origin >= cap_per_origin:
                break
            pops += 1
            current, current_pcb, depth = queue.popleft()

            if depth >= max_depth:
                continue
            if self.G.nodes[current]['isd'] != isd:
                continue

            path_sig = tuple(h['as'] for h in current_pcb.path)

            for neighbor in self.G.successors(current):
                if neighbor in path_sig:
                    continue
                if self.G.nodes[neighbor]['isd'] != isd:
                    continue

                edge_data = self.G.get_edge_data(current, neighbor)
                if not edge_data:
                    continue

                # Only ``parent-child`` (top-down) half-edges carry intra-ISD
                # beacons. Skip ``child-parent`` (bottom-up) and ``peer``.
                valid_edge = None
                for _, edge_info in edge_data.items():
                    if edge_info.get('type') == 'parent-child':
                        valid_edge = edge_info
                        break
                if not valid_edge:
                    continue

                new_path_sig = path_sig + (neighbor,)
                if new_path_sig in visited_paths:
                    continue
                visited_paths.add(new_path_sig)

                new_pcb = current_pcb.copy()
                new_pcb.path[-1]['egress'] = valid_edge.get('u_if')
                new_pcb.path.append({
                    'as': neighbor,
                    'ingress': valid_edge.get('v_if'),
                    'egress': None,
                })

                if self.G.nodes[neighbor]['role'] == 'non-core':
                    if self._register_intra_segment(new_pcb, isd, 'down', stats):
                        segments_for_origin += 1
                    reached_non_cores.add(neighbor)

                queue.append((neighbor, new_pcb, depth + 1))

                if segments_for_origin >= cap_per_origin:
                    break

        return reached_non_cores
    
    def _register_intra_segment(
        self, pcb: PCB, isd: int, seg_type: str, stats: Dict
    ) -> bool:
        """Register an intra-ISD segment + its symmetric up segment.

        Returns True if a new segment was registered (False on dedupe / cap hit).
        """
        if len(self.segments['down'][isd]) >= self.max_intra_segments_per_isd:
            self._intra_segment_cap_hit = True
            if not self._intra_seg_cap_warned:
                self._intra_seg_cap_warned = True
                print(
                    "    Warning: intra-ISD segment cap reached "
                    f"({self.max_intra_segments_per_isd} down segments per ISD)."
                )
            return False

        down_segment = {
            'type': 'down',
            'src': pcb.originator,
            'dst': pcb.path[-1]['as'],
            'hops': [h.copy() for h in pcb.path],
            'path': [h['as'] for h in pcb.path],
            'isd': isd,
            'segment_id': pcb.segment_id + "_down",
            'timestamp': time.time(),
        }
        dkey = (
            down_segment['src'],
            down_segment['dst'],
            tuple(down_segment['path']),
        )
        if dkey in self._intra_down_keys_by_isd[isd]:
            return False
        self._intra_down_keys_by_isd[isd].add(dkey)

        self.segments['down'][isd].append(down_segment)
        stats['down_segments'] += 1

        up_hops = []
        for i in range(len(pcb.path) - 1, -1, -1):
            h = pcb.path[i]
            up_hops.append(
                {
                    'as': h['as'],
                    'ingress': h['egress'],
                    'egress': h['ingress'],
                }
            )

        up_segment = {
            'type': 'up',
            'src': pcb.path[-1]['as'],
            'dst': pcb.originator,
            'hops': up_hops,
            'path': [h['as'] for h in up_hops],
            'isd': isd,
            'segment_id': pcb.segment_id + "_up",
            'timestamp': time.time(),
        }
        self.segments['up'][isd].append(up_segment)
        stats['up_segments'] += 1
        return True
    
    def _convert_to_legacy_format(self) -> Dict:
        """Convert to format expected by PathFinder"""
        legacy = {
            'core_segments': self.segments['core'],
            'up_segments_by_isd': dict(self.segments['up']),
            'down_segments_by_isd': dict(self.segments['down'])
        }
        
        return legacy


def run_corrected_simulation(topology_path: Path, output_dir: Path):
    """Run the SCION beacon simulation and print summary stats."""
    simulator = CorrectedBeaconSimulator()
    segment_store, stats = simulator.simulate(topology_path, output_dir)

    print("\n=== Simulation Complete ===")
    print("\nTotal segments discovered:")
    print(f"  Core: {stats['totals']['core_segments']}")
    print(f"  Up: {stats['totals']['up_segments']}")
    print(f"  Down: {stats['totals']['down_segments']}")

    return segment_store, stats