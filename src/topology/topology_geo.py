"""
Shared geographic helpers for topology generation (BRITE and top-down).

Centralises: k-means ISD assignment (same idea as ``BRITE2SCIONConverter``),
inter-ISD **core ring** connectivity, edge latency from plane distance, and
geography PNG export used by both pipelines.
"""

from __future__ import annotations

import math
import random as random_mod
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans


def euclidean_latency(x1: float, y1: float, x2: float, y2: float) -> float:
    """Map Euclidean separation to a float latency (ms-style units)."""
    dist = float(math.hypot(x1 - x2, y1 - y2))
    return max(0.5, 0.02 * dist)


def assign_isds_kmeans_coordinates(
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    n_isds: int,
    *,
    random_state: int = 42,
) -> Dict[int, int]:
    """
    Assign each node index ``i`` to an ISD ``0 .. n_isds-1`` using k-means on
    ``(x, y)``, matching the BRITE converter behaviour.

    Cluster ids are **renumbered** deterministically by sorting cluster
    centroids on ``(x, y)`` so ``isd_id`` order is stable across runs.
    """
    n = len(xs)
    if n_isds <= 1 or n == 0:
        return {i: 0 for i in range(n)}

    coords = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    km = KMeans(
        n_clusters=n_isds,
        random_state=int(random_state),
        n_init=10,
    )
    raw_labels = km.fit_predict(coords)
    centroids = km.cluster_centers_
    order = sorted(
        range(n_isds),
        key=lambda c: (float(centroids[c][0]), float(centroids[c][1])),
    )
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return {i: remap[int(raw_labels[i])] for i in range(n)}


def closest_core_pair_across_isds(
    G: nx.Graph,
    cores_a: Sequence[int],
    cores_b: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """Return the core pair (one from each list) with minimum squared distance."""
    if not cores_a or not cores_b:
        return None
    best: Optional[Tuple[int, int]] = None
    best_d = float("inf")
    for u in cores_a:
        du = G.nodes[u]
        xu, yu = float(du["x"]), float(du["y"])
        for v in cores_b:
            dv = G.nodes[v]
            d = (xu - float(dv["x"])) ** 2 + (yu - float(dv["y"])) ** 2
            if d < best_d:
                best_d = d
                best = (int(u), int(v))
    return best


def add_inter_isd_core_ring_edges(
    G: nx.Graph,
    isd_of: Mapping[int, int],
    core_ases: Set[int],
    *,
    bandwidth: float = 10_000.0,
) -> int:
    """
    Connect every ISD to the next on a **ring** (same pattern as
    ``BRITE2SCIONConverter._ensure_core_connectivity``): for sorted ISD ids
    ``d_0, …, d_{k-1}``, add one ``CORE`` edge between the geographically closest
    core pair for each ``(d_i, d_{i+1 mod k})``.

    Guarantees the **union of cores** is connected across ISDs when each ISD
    has at least one core and intra-ISD core subgraphs are already connected.
    """
    isd_cores: Dict[int, List[int]] = {}
    for c in core_ases:
        isd_cores.setdefault(int(isd_of[c]), []).append(int(c))

    isds = sorted(isd_cores.keys())
    if len(isds) < 2:
        return 0

    added = 0
    for i in range(len(isds)):
        a, b = isds[i], isds[(i + 1) % len(isds)]
        pair = closest_core_pair_across_isds(G, isd_cores[a], isd_cores[b])
        if pair is None:
            continue
        u, v = pair
        if u == v or G.has_edge(u, v):
            continue
        du, dv = G.nodes[u], G.nodes[v]
        lat = euclidean_latency(
            float(du["x"]), float(du["y"]), float(dv["x"]), float(dv["y"])
        )
        G.add_edge(u, v, type="CORE", latency=lat, bandwidth=bandwidth)
        added += 1
    return added


def select_cores_by_centroid_proximity(
    xs: np.ndarray,
    ys: np.ndarray,
    isd_of: Mapping[int, int],
    n_isds: int,
    rng: random_mod.Random,
) -> Set[int]:
    """
    Pick **5–10%** of ASes per ISD (at least one), choosing those **closest** to
    the ISD's geographic centroid (stable alternative to random sampling when
    the AS graph has not yet been wired).
    """
    core: Set[int] = set()
    n = len(xs)
    for isd in range(n_isds):
        members = [i for i in range(n) if int(isd_of[i]) == isd]
        if not members:
            continue
        cx = float(np.mean([xs[i] for i in members]))
        cy = float(np.mean([ys[i] for i in members]))
        ranked = sorted(
            members,
            key=lambda i: (float(xs[i]) - cx) ** 2 + (float(ys[i]) - cy) ** 2,
        )
        frac = rng.uniform(0.05, 0.10)
        n_core = max(1, int(round(len(members) * frac)))
        n_core = min(n_core, len(members), max(1, int(0.10 * len(members))))
        for i in ranked[:n_core]:
            core.add(i)
    return core


def save_topology_geography_png(
    out_path: Path,
    G: nx.Graph,
    core_ases: Set[int],
    title: str,
    *,
    xy_axis_label: str = "BRITE layout",
) -> None:
    """Write a geographic snapshot of ``G`` (expects ``x``/``y`` on nodes)."""
    import matplotlib.pyplot as plt

    pos: Dict[int, Tuple[float, float]] = {}
    for n, d in G.nodes(data=True):
        if "x" in d and "y" in d:
            pos[int(n)] = (float(d["x"]), float(d["y"]))
    missing = [n for n in G.nodes() if int(n) not in pos]
    if missing:
        sub = G.subgraph(missing).copy()
        if sub.number_of_nodes() > 0:
            spr = nx.spring_layout(sub, seed=42)
            for n in missing:
                ni = int(n)
                if ni in spr:
                    pos[ni] = (float(spr[n][0]), float(spr[n][1]))

    edge_color_map = {
        "CORE": "#e74c3c",
        "PARENT_CHILD": "#3498db",
        "CHILD_PARENT": "#85c1e9",
        "PEER": "#27ae60",
        "peer": "#27ae60",
    }

    fig, ax = plt.subplots(figsize=(11, 9))
    for u, v in G.edges():
        d = G[u][v]
        et = str(d.get("type", ""))
        ec = edge_color_map.get(et, "#bdc3c7")
        w = 2.0 if et in ("CORE", "PEER", "peer") else 1.0
        if u in pos and v in pos:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=ec,
                linewidth=w,
                alpha=0.65,
                zorder=1,
            )

    non_core = [n for n in G.nodes() if int(n) not in core_ases]
    core_list = [n for n in G.nodes() if int(n) in core_ases]

    if non_core:
        ax.scatter(
            [pos[int(n)][0] for n in non_core if int(n) in pos],
            [pos[int(n)][1] for n in non_core if int(n) in pos],
            s=55,
            c="#7f8c8d",
            zorder=3,
            label="Non-core AS",
        )
    if core_list:
        ax.scatter(
            [pos[int(n)][0] for n in core_list if int(n) in pos],
            [pos[int(n)][1] for n in core_list if int(n) in pos],
            s=140,
            c="#2c3e50",
            marker="s",
            zorder=4,
            label="Core AS",
        )

    if G.number_of_nodes() <= 100:
        for n in G.nodes():
            ni = int(n)
            if ni not in pos:
                continue
            ax.annotate(
                str(ni),
                pos[ni],
                fontsize=6,
                ha="center",
                va="center",
                color="white" if ni in core_ases else "black",
                zorder=5,
            )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"x ({xy_axis_label})")
    ax.set_ylabel(f"y ({xy_axis_label})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    if core_list or non_core:
        ax.legend(loc="upper right", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
