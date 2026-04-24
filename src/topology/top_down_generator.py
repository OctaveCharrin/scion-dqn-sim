"""
Pure-Python top-down SCION-style topology generator.

This module builds a synthetic AS-level graph (``networkx.Graph``) with ISDs,
core meshes, hierarchical parent/child edges, and geographic peering. It is
meant as a BRITE-free alternative that matches the evaluation pipeline dict
shape produced by ``BRITE2SCIONConverter.convert_brite_file``:

    {"graph": G, "isds": [...], "core_ases": set(...)}

ISDs are assigned by **k-means on (x, y)** (same approach as the BRITE
converter). Inter-ISD **core ring** links mirror ``_ensure_core_connectivity``
so the global graph stays connected when combined with intra-ISD structure.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from src.topology.topology_geo import (
    add_inter_isd_core_ring_edges,
    assign_isds_kmeans_coordinates,
    euclidean_latency,
    save_topology_geography_png,
    select_cores_by_centroid_proximity,
)


class TopDownSCIONGenerator:
    """
    Build a SCION-like topology from the top (ISDs, cores) downward.

    The graph is **undirected** (``networkx.Graph``). Each logical parent/child
    relationship is stored as a **single** edge; we label that edge with either
    ``PARENT_CHILD`` (tree parent is a **core** AS) or ``CHILD_PARENT`` (tree
    parent is **non-core**), so both SCION-style type strings appear in the
    graph without duplicating physical links.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate(
        self,
        n_isds: int = 3,
        n_nodes: int = 100,
        *,
        geographic_peering_distance_cap: Optional[float] = None,
        geographic_peering_probability: Optional[float] = None,
        additional_random_peering_max_links: int = 0,
        additional_random_peering_seed: Optional[int] = None,
        plot_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate a topology and return the pipeline-compatible dict.

        Parameters
        ----------
        plot_dir:
            If set, writes three PNG snapshots (see module docstring above).
        """
        if n_isds < 1:
            raise ValueError("n_isds must be >= 1")
        if n_nodes < n_isds:
            raise ValueError("n_nodes must be >= n_isds so each ISD has at least one AS")

        plot_dir = Path(plot_dir) if plot_dir is not None else None
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)

        km_seed = int(self._seed) if self._seed is not None else 42

        G = nx.Graph()
        xs = self._np_rng.uniform(0.0, 1000.0, size=n_nodes)
        ys = self._np_rng.uniform(0.0, 1000.0, size=n_nodes)

        isd_of = assign_isds_kmeans_coordinates(
            xs, ys, n_isds, random_state=km_seed
        )

        for n in range(n_nodes):
            G.add_node(
                n,
                isd=int(isd_of[n]),
                role="non-core",
                x=float(xs[n]),
                y=float(ys[n]),
            )

        core_ases = select_cores_by_centroid_proximity(
            xs, ys, isd_of, n_isds, self._rng
        )
        for c in core_ases:
            G.nodes[c]["role"] = "core"

        _conv_holder: list[Any] = [None]

        def _brite_converter() -> Any:
            if _conv_holder[0] is None:
                from src.topology.brite2scion_converter import BRITE2SCIONConverter

                _conv_holder[0] = BRITE2SCIONConverter()
            return _conv_holder[0]

        if plot_dir is not None:
            G_layout = nx.Graph()
            for n in G.nodes():
                G_layout.add_node(n, **dict(G.nodes[n]))
            save_topology_geography_png(
                plot_dir / "step1_top_down_layout.png",
                G_layout,
                set(core_ases),
                "Step 1: Top-down — k-means ISDs, cores (no edges)",
                xy_axis_label="synthetic plane",
            )

        self._add_intra_isd_core_mesh(G, isd_of, core_ases)
        add_inter_isd_core_ring_edges(G, isd_of, core_ases)

        self._attach_non_core_hierarchy(G, isd_of, core_ases, xs, ys)

        if plot_dir is not None:
            save_topology_geography_png(
                plot_dir / "step2_top_down_hierarchy.png",
                G.copy(),
                set(core_ases),
                "Step 2: Top-down — core mesh + ring + hierarchy (no peering)",
                xy_axis_label="synthetic plane",
            )

        self._add_geographic_peering(
            G,
            xs,
            ys,
            distance_cap=geographic_peering_distance_cap,
            probability=geographic_peering_probability,
        )

        if additional_random_peering_max_links > 0:
            ar_seed = (
                additional_random_peering_seed
                if additional_random_peering_seed is not None
                else (self._seed if self._seed is not None else 42)
            )
            _brite_converter().add_random_peering_links(
                G,
                rng=np.random.default_rng(int(ar_seed)),
                max_links=int(additional_random_peering_max_links),
            )

        if plot_dir is not None:
            save_topology_geography_png(
                plot_dir / "step3_top_down_peering.png",
                G,
                set(core_ases),
                "Step 3: Top-down — + geographic & extra random PEER edges",
                xy_axis_label="synthetic plane",
            )
            print(
                f"\nWrote top-down topology PNGs under {plot_dir}: "
                "step1_top_down_layout.png, step2_top_down_hierarchy.png, "
                "step3_top_down_peering.png"
            )

        isds: List[Dict[str, Any]] = []
        for isd_id in range(n_isds):
            member_ases = sorted(
                nid for nid in range(n_nodes) if isd_of[nid] == isd_id
            )
            isds.append({"isd_id": int(isd_id), "member_ases": member_ases})

        return {"graph": G, "isds": isds, "core_ases": core_ases}

    def _add_intra_isd_core_mesh(
        self,
        G: nx.Graph,
        isd_of: Dict[int, int],
        core_ases: Set[int],
    ) -> None:
        """Fully connect all core ASes that belong to the same ISD (CORE edges)."""
        cores_by_isd: Dict[int, List[int]] = {}
        for c in core_ases:
            cores_by_isd.setdefault(isd_of[c], []).append(c)

        for _, cores in cores_by_isd.items():
            if len(cores) < 2:
                continue
            for i, u in enumerate(cores):
                for v in cores[i + 1 :]:
                    self._add_core_edge(G, u, v)

    def _add_core_edge(self, G: nx.Graph, u: int, v: int) -> None:
        du, dv = G.nodes[u], G.nodes[v]
        lat = euclidean_latency(du["x"], du["y"], dv["x"], dv["y"])
        G.add_edge(u, v, type="CORE", latency=lat, bandwidth=10_000.0)

    def _attach_non_core_hierarchy(
        self,
        G: nx.Graph,
        isd_of: Dict[int, int],
        core_ases: Set[int],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        n_isds = max(isd_of.values(), default=-1) + 1

        for isd_id in range(n_isds):
            isd_noncore = [
                n for n in G.nodes if isd_of[n] == isd_id and n not in core_ases
            ]
            self._rng.shuffle(isd_noncore)

            attached_in_isd: Set[int] = {
                nid for nid in G.nodes if isd_of[nid] == isd_id and nid in core_ases
            }

            for n in isd_noncore:
                candidates = [p for p in attached_in_isd if p != n]
                if not candidates:
                    raise RuntimeError(
                        f"No attachment candidates for node {n} in ISD {isd_id}"
                    )

                weights = [max(1, int(G.degree(p))) for p in candidates]
                parent = self._weighted_choice(candidates, weights)

                xn, yn = float(xs[n]), float(ys[n])
                xp, yp = float(xs[parent]), float(ys[parent])
                lat = euclidean_latency(xn, yn, xp, yp)

                etype = "PARENT_CHILD" if parent in core_ases else "CHILD_PARENT"
                G.add_edge(n, parent, type=etype, latency=lat, bandwidth=1_000.0)
                attached_in_isd.add(n)

    def _weighted_choice(self, items: List[int], weights: List[int]) -> int:
        total = sum(weights)
        r = self._rng.uniform(0.0, total)
        upto = 0.0
        for item, w in zip(items, weights):
            upto += float(w)
            if upto >= r:
                return item
        return items[-1]

    def _add_geographic_peering(
        self,
        G: nx.Graph,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        distance_cap: Optional[float] = None,
        probability: Optional[float] = None,
    ) -> None:
        non_core = [n for n in G.nodes if G.nodes[n]["role"] == "non-core"]
        if len(non_core) < 2:
            return

        n = G.number_of_nodes()
        if distance_cap is None:
            dist_cap = 1000.0 / math.sqrt(max(n, 2))
        else:
            dist_cap = float(distance_cap)

        if probability is None:
            p_peer = self._rng.uniform(0.10, 0.15)
        else:
            p_peer = float(probability)

        for i, u in enumerate(non_core):
            xu, yu = float(xs[u]), float(ys[u])
            for v in non_core[i + 1 :]:
                if G.has_edge(u, v):
                    continue
                xv, yv = float(xs[v]), float(ys[v])
                dist = float(math.hypot(xu - xv, yu - yv))
                if dist > dist_cap:
                    continue
                if self._rng.random() > p_peer:
                    continue
                lat = euclidean_latency(xu, yu, xv, yv)
                G.add_edge(u, v, type="PEER", latency=lat, bandwidth=500.0)
