"""
BRITE to SCION topology converter

Converts BRITE topologies to SCION format with:
- ISD assignment using k-means clustering
- Core AS selection based on degree centrality
- Link type classification
- Interface ID management
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.topology.topology_geo import (
    assign_isds_kmeans_coordinates,
    euclidean_latency,
    save_topology_geography_png,
)


class BRITE2SCIONConverter:
    """Convert BRITE topologies to SCION format"""

    def __init__(self, n_isds: int = 3, core_ratio: float = 0.075):
        """
        Args:
            n_isds: Number of ISDs to create (automatically reduced to 1 for topologies < 200 ASes)
            core_ratio: Fraction of ASes to designate as core per ISD (default 7.5%, bounded to 5-10%)
        """
        self.n_isds = n_isds
        self.original_n_isds = n_isds  # Store original value
        self.core_ratio = core_ratio

    def convert_brite_file(
        self,
        brite_file: Path,
        *,
        plot_dir: Optional[Path] = None,
        extra_peering_max_links: Optional[int] = None,
        extra_peering_seed: Optional[int] = None,
        prune_cross_isd_noncore_fraction: float = 0.0,
        prune_cross_isd_noncore_seed: Optional[int] = None,
    ) -> Dict:
        """
        Load a BRITE ``.brite`` export, run SCION assignment/classification, optional
        random peering densification, and return ``graph`` + ``isds`` + ``core_ases``.

        If ``plot_dir`` is set, writes three PNG snapshots:

        1. ``step1_vanilla_brite.png`` — positions + raw BRITE edges only.
        2. ``step2_scion_enhanced.png`` — after ISD/core/virtual/dense links + classification.
        3. ``step3_peering_enhanced.png`` — after extra random PEER links.

        Parameters
        ----------
        extra_peering_max_links:
            Passed to ``add_random_peering_links(..., max_links=...)``. ``None`` keeps
            the built-in cap (function of ``n``).
        extra_peering_seed:
            Seed for the NumPy RNG used during extra peering. ``None`` uses ``42``.
        prune_cross_isd_noncore_fraction:
            If ``> 0``, after link classification removes approximately this
            fraction of edges whose **both** endpoints are non-core and lie in
            **different** ISDs (typical BRITE/dense-helper artefacts). Does not
            remove edges touching a core AS. Runs **before** random PEER
            injection so explicit peering still reflects the intended policy.
        prune_cross_isd_noncore_seed:
            RNG seed for the prune shuffle. ``None`` reuses ``extra_peering_seed``
            resolution (global ``seed`` / ``42``).
        """
        brite_file = Path(brite_file)
        plot_dir = Path(plot_dir) if plot_dir is not None else None
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)

        G, node_attrs = self._read_brite_edges(brite_file)
        for n in G.nodes():
            G.nodes[n]["x"] = float(node_attrs[n]["x"])
            G.nodes[n]["y"] = float(node_attrs[n]["y"])

        if plot_dir is not None:
            self._save_topology_step_png(
                plot_dir / "step1_vanilla_brite.png",
                G,
                set(),
                "Step 1: Vanilla BRITE (layout + physical edges)",
            )

        peer_seed = 42 if extra_peering_seed is None else int(extra_peering_seed)
        # Single, seeded RNG drives every randomized step (core wiring, dense
        # links, prune, peering). Module-level ``np.random`` was previously used
        # in ``_add_dense_connections`` which made step 1 non-reproducible.
        struct_rng = np.random.default_rng(peer_seed)

        isd_assignment = self._assign_isds(G, node_attrs)
        core_ases = self._select_core_ases(G, isd_assignment)
        self._ensure_core_connectivity(G, core_ases, isd_assignment)
        self._ensure_multi_parent_connectivity(G, core_ases, isd_assignment)
        self._add_dense_connections(G, core_ases, isd_assignment, rng=struct_rng)
        link_types = self._classify_links(G, core_ases, isd_assignment)

        for n in G.nodes():
            G.nodes[n]["isd"] = int(isd_assignment[n])
            G.nodes[n]["role"] = "core" if int(n) in core_ases else "non-core"

        for u, v in G.edges():
            lt = link_types.get((u, v)) or link_types.get((v, u))
            if lt is None:
                # Cross-ISD non-core BRITE-physical edges: classify as PEER.
                # ``prune_cross_isd_noncore_fraction`` (below) thins these out
                # before extra random peering injection.
                lt = "peer"
            edge = G[u][v]
            # Prefer an explicit ``latency`` set by the synthetic edge helpers
            # (core mesh, multi-parent, dense, peering). For BRITE-original
            # edges we recompute from plane distance so units stay consistent
            # across all edges (BRITE's ``delay`` field is on an abstract scale
            # that depends on the routing model and is not in milliseconds).
            if "latency" in edge:
                lat = float(edge["latency"])
            else:
                u_x = float(G.nodes[u].get("x", 0.0))
                u_y = float(G.nodes[u].get("y", 0.0))
                v_x = float(G.nodes[v].get("x", 0.0))
                v_y = float(G.nodes[v].get("y", 0.0))
                lat = euclidean_latency(u_x, u_y, v_x, v_y)
            bw = float(edge.get("bandwidth", 10.0))
            edge["type"] = lt.upper().replace("-", "_")
            edge["latency"] = lat
            edge["bandwidth"] = bw

        if prune_cross_isd_noncore_fraction and prune_cross_isd_noncore_fraction > 0:
            pseed = (
                int(prune_cross_isd_noncore_seed)
                if prune_cross_isd_noncore_seed is not None
                else peer_seed
            )
            p_rng = np.random.default_rng(pseed)
            n_pruned = self._prune_cross_isd_noncore_edges(
                G,
                core_ases,
                isd_assignment,
                float(prune_cross_isd_noncore_fraction),
                p_rng,
            )
            print(
                f"\nPruned {n_pruned} cross-ISD non-core edge(s) "
                f"(target fraction {float(prune_cross_isd_noncore_fraction):.2%})"
            )

        if plot_dir is not None:
            self._save_topology_step_png(
                plot_dir / "step2_scion_enhanced.png",
                G,
                set(core_ases),
                "Step 2: SCION enhancements (ISD, core, dense links, classified types)",
            )

        peer_rng = np.random.default_rng(peer_seed)
        n_peer = self.add_random_peering_links(
            G, rng=peer_rng, max_links=extra_peering_max_links
        )
        print(f"\nAdded {n_peer} random PEER link(s) for dense connectivity")

        if plot_dir is not None:
            self._save_topology_step_png(
                plot_dir / "step3_peering_enhanced.png",
                G,
                set(core_ases),
                f"Step 3: + random peering ({n_peer} new PEER edges)",
            )

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

    def add_random_peering_links(
        self,
        G: nx.Graph,
        *,
        rng: Optional[np.random.Generator] = None,
        max_links: Optional[int] = None,
    ) -> int:
        """Add random bidirectional-style PEER edges (single undirected edge in ``Graph``).

        Nodes must already have ``isd`` set. Matches the former logic in
        ``evaluation/01_generate_topology.py`` (prefer inter-ISD peering).
        """
        rng = rng or np.random.default_rng(42)
        nodes = [int(n) for n in G.nodes()]
        n = len(nodes)
        if n < 2:
            return 0
        cap = max_links if max_links is not None else min(75, max(2, n * n // 4))
        interface_id = 1000
        added = 0
        attempts = 0
        max_attempts = max(10000, cap * 500)
        while added < cap and attempts < max_attempts:
            attempts += 1
            src, dst = int(rng.choice(nodes)), int(rng.choice(nodes))
            if src == dst or G.has_edge(src, dst):
                continue

            # SCION peer links typically materialize at IXPs / local colos so we
            # bias acceptance toward geographically close pairs. The exponential
            # decay constant is expressed as a fraction of the layout extent so
            # the heuristic adapts to BRITE (HS=1000) and top-down (1000x1000)
            # planes without retuning.
            pos1 = (G.nodes[src].get("x", 0), G.nodes[src].get("y", 0))
            pos2 = (G.nodes[dst].get("x", 0), G.nodes[dst].get("y", 0))
            geo_dist = float(
                np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
            )
            decay = self._estimate_layout_extent(G) * 0.25
            prob = float(np.exp(-geo_dist / max(decay, 1.0)))
            if rng.random() > prob:
                continue

            lat = euclidean_latency(
                float(pos1[0]), float(pos1[1]), float(pos2[0]), float(pos2[1])
            )
            G.add_edge(
                src,
                dst,
                src_if=interface_id,
                dst_if=interface_id + 1,
                type="PEER",
                bandwidth=float(rng.uniform(5000, 10000)),
                latency=lat,
            )
            interface_id += 2
            added += 1
        return added

    @staticmethod
    def _estimate_layout_extent(G: nx.Graph) -> float:
        """Approximate the size of the (x, y) plane used by a topology layout.

        Used to scale geographic decay constants without hard-coding a BRITE
        ``HS`` value. Returns at least 1.0 to keep the exponent finite.
        """
        xs: List[float] = []
        ys: List[float] = []
        for _, data in G.nodes(data=True):
            if "x" in data and "y" in data:
                xs.append(float(data["x"]))
                ys.append(float(data["y"]))
        if not xs or not ys:
            return 1000.0
        return max(1.0, max(max(xs) - min(xs), max(ys) - min(ys)))

    def _prune_cross_isd_noncore_edges(
        self,
        G: nx.Graph,
        core_ases: Set[int],
        isd_assignment: Dict[int, int],
        fraction: float,
        rng: np.random.Generator,
    ) -> int:
        """
        Remove a random subset of edges between two **non-core** ASes in
        **different** ISDs.

        BRITE plus dense-connection helpers often leave many such edges; trimming
        a fraction mimics selective settlement / fewer direct non-core ties
        across ISDs while **explicit** PEER injection (later) still models IXPs.
        """
        if fraction <= 0:
            return 0
        fraction = min(1.0, float(fraction))
        cands: List[Tuple[int, int]] = []
        for u, v in list(G.edges()):
            if u in core_ases or v in core_ases:
                continue
            if int(isd_assignment[u]) == int(isd_assignment[v]):
                continue
            cands.append((int(u), int(v)))
        if not cands:
            return 0
        n_remove = int(round(len(cands) * fraction))
        n_remove = min(len(cands), max(0, n_remove))
        order = rng.permutation(len(cands))
        removed = 0
        for j in range(n_remove):
            u, v = cands[int(order[j])]
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                removed += 1
        return removed

    def _save_topology_step_png(
        self,
        out_path: Path,
        G: nx.Graph,
        core_ases: Set[int],
        title: str,
        *,
        xy_axis_label: str = "BRITE layout",
    ) -> None:
        """Write a quick geographic snapshot of ``G`` (expects ``x``/``y`` on nodes)."""
        save_topology_geography_png(
            out_path, G, core_ases, title, xy_axis_label=xy_axis_label
        )

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
                        # BRITE's ``delay`` column is on an abstract scale that
                        # depends on the routing model. We derive latency from
                        # the layout distance (``euclidean_latency``) once we
                        # have node coordinates; here we keep the raw value as
                        # a fallback in case coords are missing.
                        delay = float(parts[4])
                        bw = float(parts[5])
                        G.add_edge(u, v, bandwidth=bw, delay=delay, length=dist)

        return G, node_attrs

    def _assign_isds(self, G: nx.Graph, node_attrs: Dict) -> Dict[int, int]:
        """Assign nodes to ISDs using k-means on geographic coordinates."""
        nodes = sorted(G.nodes())
        if self.n_isds == 1:
            return {node: 0 for node in nodes}
        xs = np.array([float(node_attrs[n]["x"]) for n in nodes])
        ys = np.array([float(node_attrs[n]["y"]) for n in nodes])
        by_index = assign_isds_kmeans_coordinates(
            xs, ys, self.n_isds, random_state=42
        )
        return {nodes[i]: by_index[i] for i in range(len(nodes))}

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

            # Calculate number of core ASes
            n_core = max(1, int(len(nodes) * self.core_ratio))

            # # Ensure we don't exceed node count even with rounding
            max_core = max(1, int(len(nodes)))
            n_core = min(n_core, max_core)

            # Sort by degree and select top nodes
            sorted_nodes = sorted(nodes, key=lambda n: degrees[n], reverse=True)
            selected_core = sorted_nodes[:n_core]
            core_ases.update(selected_core)

            print(
                f"  ISD {isd}: {len(nodes)} ASes, {n_core} core ASes ({n_core / len(nodes) * 100:.1f}%)"
            )

        return core_ases

    def _ensure_core_connectivity(
        self, G: nx.Graph, core_ases: set, isd_assignment: Dict
    ):
        """Ensure core ASes form a connected subgraph per ISD, and connect ISDs globally."""
        isd_cores = {}
        for node in core_ases:
            isd = isd_assignment[node]
            if isd not in isd_cores:
                isd_cores[isd] = []
            isd_cores[isd].append(node)

        # 1. Intra-ISD Core Connectivity (Full mesh per ISD is standard for SCION)
        for isd, cores in isd_cores.items():
            print(f"  Ensuring core connectivity for ISD {isd}...")
            added = 0
            n_cores = len(cores)
            for i in range(n_cores):
                for j in range(i + 1, n_cores):
                    u = cores[i]
                    v = cores[j]
                    if not G.has_edge(u, v):
                        lat = euclidean_latency(
                            float(G.nodes[u].get("x", 0.0)),
                            float(G.nodes[u].get("y", 0.0)),
                            float(G.nodes[v].get("x", 0.0)),
                            float(G.nodes[v].get("y", 0.0)),
                        )
                        G.add_edge(
                            u,
                            v,
                            bandwidth=100000.0,
                            latency=lat,
                            virtual=True,
                        )
                        print(f"    Added intra-ISD core link: {u} <-> {v}")
                        added += 1
            if added == 0:
                print("    Core ASes already fully connected")

        # 2. Inter-ISD Core Connectivity (Global TRC connectivity)
        isds = sorted(list(isd_cores.keys()))
        if len(isds) > 1:
            print("  Ensuring inter-ISD core connectivity...")
            for i in range(len(isds)):
                isd_a = isds[i]
                isd_b = isds[(i + 1) % len(isds)]

                # Find closest pair of cores between isd_a and isd_b for realistic linkage
                best_pair = None
                min_dist = float("inf")
                for u in isd_cores[isd_a]:
                    for v in isd_cores[isd_b]:
                        u_pos = (G.nodes[u].get("x", 0), G.nodes[u].get("y", 0))
                        v_pos = (G.nodes[v].get("x", 0), G.nodes[v].get("y", 0))
                        dist = (u_pos[0] - v_pos[0]) ** 2 + (u_pos[1] - v_pos[1]) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (u, v)

                if best_pair:
                    u, v = best_pair
                    if not G.has_edge(u, v):
                        lat = euclidean_latency(
                            float(G.nodes[u].get("x", 0.0)),
                            float(G.nodes[u].get("y", 0.0)),
                            float(G.nodes[v].get("x", 0.0)),
                            float(G.nodes[v].get("y", 0.0)),
                        )
                        G.add_edge(
                            u,
                            v,
                            bandwidth=50000.0,
                            latency=lat,
                            virtual=True,
                        )
                        print(f"    Added inter-ISD core link: {u} <-> {v}")

    def _ensure_multi_parent_connectivity(
        self, G: nx.Graph, core_ases: set, isd_assignment: Dict
    ):
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
                dist_to_core = self._distance_to_core(
                    node, G, core_ases, isd_assignment, isd
                )

                if dist_to_core > 2:  # Far from core, needs better connectivity
                    # Find potential parents (ASes closer to core)
                    potential_parents = []

                    for other in nodes:
                        if other == node or other in core_ases:
                            continue

                        other_dist = self._distance_to_core(
                            other, G, core_ases, isd_assignment, isd
                        )
                        if (
                            other_dist < dist_to_core - 1
                        ):  # Significantly closer to core
                            # Check if not already connected
                            if not G.has_edge(node, other):
                                potential_parents.append((other, other_dist))

                    # Sort by distance to core
                    potential_parents.sort(key=lambda x: x[1])

                    # Add connections to 1-2 closest potential parents
                    for i, (parent, _) in enumerate(potential_parents[:2]):
                        # Calculate geographic distance
                        node_pos = (
                            G.nodes[node].get("x", 0),
                            G.nodes[node].get("y", 0),
                        )
                        parent_pos = (
                            G.nodes[parent].get("x", 0),
                            G.nodes[parent].get("y", 0),
                        )
                        geo_dist = np.sqrt(
                            (node_pos[0] - parent_pos[0]) ** 2
                            + (node_pos[1] - parent_pos[1]) ** 2
                        )

                        # Only add if geographically reasonable
                        if geo_dist < 400:  # Reasonable distance threshold
                            lat = euclidean_latency(
                                float(node_pos[0]),
                                float(node_pos[1]),
                                float(parent_pos[0]),
                                float(parent_pos[1]),
                            )
                            G.add_edge(
                                node,
                                parent,
                                bandwidth=1000.0,
                                latency=lat,
                                virtual=True,
                            )
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
                            neighbor_dist = self._distance_to_core(
                                neighbor, G, core_ases, isd_assignment, isd
                            )
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
                                other_dist = self._distance_to_core(
                                    other, G, core_ases, isd_assignment, isd
                                )

                            if other_dist < dist_to_core and not G.has_edge(
                                node, other
                            ):
                                # Check geographic distance
                                node_pos = (
                                    G.nodes[node].get("x", 0),
                                    G.nodes[node].get("y", 0),
                                )
                                other_pos = (
                                    G.nodes[other].get("x", 0),
                                    G.nodes[other].get("y", 0),
                                )
                                geo_dist = np.sqrt(
                                    (node_pos[0] - other_pos[0]) ** 2
                                    + (node_pos[1] - other_pos[1]) ** 2
                                )

                                candidates.append((other, other_dist, geo_dist))

                        # Sort by distance to core, then geographic distance
                        candidates.sort(key=lambda x: (x[1], x[2]))

                        # Add connections to multiple candidates
                        parents_to_add = min(
                            target_parents - len(parents), len(candidates)
                        )
                        for i in range(parents_to_add):
                            if (
                                candidates[i][2] < 500
                            ):  # Increased geographic distance threshold
                                best = candidates[i][0]
                                best_pos = (
                                    G.nodes[best].get("x", 0),
                                    G.nodes[best].get("y", 0),
                                )
                                lat = euclidean_latency(
                                    float(node_pos[0]),
                                    float(node_pos[1]),
                                    float(best_pos[0]),
                                    float(best_pos[1]),
                                )
                                G.add_edge(
                                    node,
                                    best,
                                    bandwidth=2000.0,
                                    latency=lat,
                                    virtual=True,
                                )
                                added_connections += 1
                                print(
                                    f"    Added parent {len(parents) + i + 1}: {node} -> {best}"
                                )

        print(f"    Total multi-parent connections added: {added_connections}")

    def _add_dense_connections(
        self,
        G: nx.Graph,
        core_ases: set,
        isd_assignment: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Add additional cross-connections for dense topology with many paths.

        ``rng`` should be a seeded ``numpy.random.Generator`` so the densification
        is reproducible from the YAML ``seed``. A default-seeded RNG is used if
        the caller passes ``None`` (kept for backwards compatibility).
        """
        print("  Adding dense cross-connections...")
        if rng is None:
            rng = np.random.default_rng(42)
        added_connections = 0

        # Group ASes by ISD and distance from core
        as_by_distance = {}
        for isd_id in set(isd_assignment.values()):
            as_by_distance[isd_id] = {}

            for as_id, isd in isd_assignment.items():
                if isd != isd_id:
                    continue

                dist = self._distance_to_core(
                    as_id, G, core_ases, isd_assignment, isd_id
                )
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
                    for j, as2 in enumerate(ases[i + 1 :], i + 1):
                        if not G.has_edge(as1, as2):
                            # Check geographic distance
                            pos1 = (G.nodes[as1].get("x", 0), G.nodes[as1].get("y", 0))
                            pos2 = (G.nodes[as2].get("x", 0), G.nodes[as2].get("y", 0))
                            geo_dist = np.sqrt(
                                (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
                            )

                            # Add connection with probability based on distance
                            if geo_dist < 200 and rng.random() < 0.15:
                                lat = euclidean_latency(
                                    float(pos1[0]),
                                    float(pos1[1]),
                                    float(pos2[0]),
                                    float(pos2[1]),
                                )
                                G.add_edge(
                                    as1,
                                    as2,
                                    bandwidth=1000.0,
                                    latency=lat,
                                    virtual=True,
                                    cross_connect=True,
                                )
                                added_connections += 1

        # Add shortcut connections between different levels
        for isd_id, distance_groups in as_by_distance.items():
            levels = sorted(distance_groups.keys())
            for i, level1 in enumerate(levels[:-1]):
                if level1 == 0:  # Skip core level
                    continue

                for level2 in levels[i + 1 :]:
                    if level2 - level1 > 2:  # Skip if too far apart
                        continue

                    # Connect some ASes between levels
                    ases1 = distance_groups[level1]
                    ases2 = distance_groups[level2]

                    for as1 in ases1[:5]:  # Limit connections
                        for as2 in ases2[:5]:
                            if not G.has_edge(as1, as2):
                                pos1 = (
                                    G.nodes[as1].get("x", 0),
                                    G.nodes[as1].get("y", 0),
                                )
                                pos2 = (
                                    G.nodes[as2].get("x", 0),
                                    G.nodes[as2].get("y", 0),
                                )
                                geo_dist = np.sqrt(
                                    (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
                                )

                                if geo_dist < 250 and rng.random() < 0.1:
                                    lat = euclidean_latency(
                                        float(pos1[0]),
                                        float(pos1[1]),
                                        float(pos2[0]),
                                        float(pos2[1]),
                                    )
                                    G.add_edge(
                                        as1,
                                        as2,
                                        bandwidth=1000.0,
                                        latency=lat,
                                        virtual=True,
                                        shortcut=True,
                                    )
                                    added_connections += 1

        print(f"    Total dense connections added: {added_connections}")

    def _classify_links(
        self, G: nx.Graph, core_ases: set, isd_assignment: Dict
    ) -> Dict[Tuple[int, int], str]:
        """Classify links as core or parent-child (no peering links)"""
        link_types = {}

        # First, ensure all non-core ASes have a path to core
        # Build a tree structure for each ISD
        self._build_isd_trees(G, core_ases, isd_assignment)

        for u, v in G.edges():
            u_core = u in core_ases
            v_core = v in core_ases
            u_isd = isd_assignment[u]
            v_isd = isd_assignment[v]

            if u_core and v_core:
                # Both core -> core link
                link_types[(u, v)] = "core"
            elif u_core and not v_core and u_isd == v_isd:
                # Core to non-core in same ISD -> parent-child
                link_types[(u, v)] = "parent-child"
            elif not u_core and v_core and u_isd == v_isd:
                # Non-core to core in same ISD -> child-parent
                link_types[(u, v)] = "child-parent"
            elif not u_core and not v_core and u_isd == v_isd:
                # Determine parent-child direction based on distance to core
                u_dist = self._distance_to_core(u, G, core_ases, isd_assignment, u_isd)
                v_dist = self._distance_to_core(v, G, core_ases, isd_assignment, v_isd)

                if u_dist < v_dist:
                    link_types[(u, v)] = "parent-child"
                elif v_dist < u_dist:
                    link_types[(u, v)] = "child-parent"
                else:
                    # Same level - make it parent-child based on AS ID for consistency
                    if u < v:
                        link_types[(u, v)] = "parent-child"
                    else:
                        link_types[(u, v)] = "child-parent"
            # Cross-ISD non-core links are not allowed in SCION

        return link_types

    def _build_isd_trees(
        self, G: nx.Graph, core_ases: set, isd_assignment: Dict
    ) -> Dict[int, set]:
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

    def _distance_to_core(
        self, node: int, G: nx.Graph, core_ases: set, isd_assignment: Dict, isd: int
    ) -> int:
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

                # SCION routing isolation: distance mapping MUST stay within the same ISD
                if isd_assignment.get(neighbor) != isd:
                    continue

                if neighbor in core_ases:
                    return dist + 1

                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

        # If no path found, return large number
        return 999
