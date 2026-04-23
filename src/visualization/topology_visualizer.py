"""
Topology visualization tool

Creates visual representations of SCION topologies with:
- ISD boundaries and labels
- Core AS highlighting
- Link type differentiation
- Geographic layout preservation

Supports both **pickle** topologies (``nodes`` / ``edges`` DataFrames, legacy CLI
shape) and **evaluation JSON** (``scion_topology.json`` with ``graph`` +
``core_ases`` + ``isds``), via :func:`load_topology_tables`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# JSON (scion_topology.json) → DataFrame bridge
# ---------------------------------------------------------------------------


def load_scion_topology_json(path: Path) -> Tuple[nx.Graph, Set[int], List[Any]]:
    """Load evaluation-style ``scion_topology.json`` (NetworkX node-link)."""
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    graph_obj = data.get("graph")
    if not graph_obj:
        raise ValueError(f"No 'graph' key in {path}")
    G = nx.node_link_graph(graph_obj)
    core_ases = {int(x) for x in data.get("core_ases", [])}
    isds = data.get("isds", [])
    return G, core_ases, isds


def _collapse_to_simple_undirected(G: nx.Graph) -> nx.Graph:
    """One undirected edge per AS pair; keep a representative ``type`` string."""
    H = nx.Graph()
    for n, attr in G.nodes(data=True):
        H.add_node(int(n), **attr)

    priority = {
        "PEER": 0,
        "peer": 0,
        "core": 1,
        "CORE": 1,
        "parent-child": 2,
        "PARENT_CHILD": 2,
        "child-parent": 2,
        "CHILD_PARENT": 2,
    }

    def type_rank(t: str) -> int:
        return priority.get(t, 9)

    edges_seen: Dict[Tuple[int, int], List[str]] = {}

    def collect(u: int, v: int, t: str) -> None:
        a, b = (u, v) if u < v else (v, u)
        edges_seen.setdefault((a, b), []).append(t or "UNKNOWN")

    if G.is_multigraph():
        for u, v, _k, d in G.edges(keys=True, data=True):
            collect(int(u), int(v), str(d.get("type", "UNKNOWN")))
    else:
        for u, v, d in G.edges(data=True):
            collect(int(u), int(v), str(d.get("type", "UNKNOWN")))

    for (a, b), types in edges_seen.items():
        ranked = sorted(types, key=type_rank)
        chosen = ranked[0] if ranked else "UNKNOWN"
        H.add_edge(a, b, type=chosen)
    return H


def _isd_for_node(n: int, isds_meta: List[Any]) -> int:
    """Infer ISD id from ``isds`` list of dicts with ``member_ases``."""
    for block in isds_meta or []:
        if not isinstance(block, dict):
            continue
        members = block.get("member_ases") or block.get("members") or []
        try:
            mem = {int(x) for x in members}
        except (TypeError, ValueError):
            continue
        if int(n) in mem:
            return int(block.get("isd_id", block.get("isd", 0)))
    return int(0)


def json_topology_to_frames(json_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Convert ``scion_topology.json`` into ``nodes`` / ``edges`` DataFrames."""
    G_raw, core_ases, isds = load_scion_topology_json(json_path)
    H = _collapse_to_simple_undirected(G_raw)

    rows = []
    for n, d in H.nodes(data=True):
        nid = int(n)
        isd = d.get("isd")
        if isd is None:
            isd = _isd_for_node(nid, isds)
        try:
            isd = int(isd)
        except (TypeError, ValueError):
            isd = 0
        role = "core" if nid in core_ases else "non-core"
        x = float(d["x"]) if d.get("x") is not None else 0.0
        y = float(d["y"]) if d.get("y") is not None else 0.0
        rows.append(
            {
                "as_id": nid,
                "isd": isd,
                "role": role,
                "x": x,
                "y": y,
                "degree": H.degree(nid),
            }
        )
    node_df = pd.DataFrame(rows)
    if node_df.empty:
        edge_df = pd.DataFrame(columns=["u", "v", "type"])
    else:
        # Spring layout for nodes missing coordinates
        missing = node_df[(node_df["x"] == 0.0) & (node_df["y"] == 0.0)]
        if len(missing) == len(node_df) and len(node_df) > 1:
            pos = nx.spring_layout(H, seed=42)
            for i, r in node_df.iterrows():
                nid = int(r["as_id"])
                if nid in pos:
                    node_df.at[i, "x"] = float(pos[nid][0])
                    node_df.at[i, "y"] = float(pos[nid][1])
        elif len(missing) > 0:
            sub = H.subgraph(missing["as_id"].astype(int).tolist()).copy()
            if sub.number_of_nodes() > 0:
                pos = nx.spring_layout(sub, seed=42)
                for i, r in node_df.iterrows():
                    nid = int(r["as_id"])
                    if nid in missing["as_id"].values and nid in pos:
                        node_df.at[i, "x"] = float(pos[nid][0])
                        node_df.at[i, "y"] = float(pos[nid][1])

        erows = []
        for u, v, d in H.edges(data=True):
            erows.append({"u": int(u), "v": int(v), "type": str(d.get("type", "UNKNOWN"))})
        edge_df = pd.DataFrame(erows)

    meta = {"source": "json", "core_ases": sorted(core_ases), "n_isds": len(isds)}
    return node_df, edge_df, meta


def load_topology_tables(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load ``nodes`` / ``edges`` frames from a pickle topology or ``scion_topology.json``."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json_topology_to_frames(path)
    with open(path, "rb") as f:
        topology = pickle.load(f)
    node_df = topology["nodes"]
    edge_df = topology["edges"]
    meta = {"source": "pickle", "keys": list(topology.keys())}
    return node_df, edge_df, meta


class TopologyVisualizer:
    """Create visualizations of SCION topologies"""
    
    # Color schemes
    ISD_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    CORE_COLOR = '#2C3E50'
    NON_CORE_COLOR = '#95A5A6'
    
    # Link styles — keys match ``edge['type']`` from pickles *or* JSON
    # (``convert_brite_file`` uses SCREAMING_SNAKE, e.g. ``PARENT_CHILD``).
    LINK_STYLES = {
        "core": {"color": "#E74C3C", "width": 3.0, "style": "-"},
        "CORE": {"color": "#E74C3C", "width": 3.0, "style": "-"},
        "parent-child": {"color": "#3498DB", "width": 2.0, "style": "-"},
        "PARENT_CHILD": {"color": "#3498DB", "width": 2.0, "style": "-"},
        "child-parent": {"color": "#3498DB", "width": 2.0, "style": "--"},
        "CHILD_PARENT": {"color": "#3498DB", "width": 2.0, "style": "--"},
        "peer": {"color": "#27AE60", "width": 1.5, "style": ":"},
        "PEER": {"color": "#1e8449", "width": 2.0, "style": "-"},
        "UNKNOWN": {"color": "#95a5a6", "width": 1.0, "style": "-"},
    }
    DEFAULT_LINK_STYLE = {"color": "#7f8c8d", "width": 1.2, "style": "-"}
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-white')
        
    def visualize_topology(
        self,
        topology_path: Path,
        output_path: Path,
        show_labels: bool = True,
        show_grid: bool = True,
        *,
        write_extras: bool = True,
        dpi: int = 300,
    ) -> None:
        """
        Create a full dashboard: main geographic map + degree / ISD / link charts.

        ``topology_path`` may be a **pickle** topology or ``scion_topology.json``.
        """
        node_df, edge_df, _meta = load_topology_tables(Path(topology_path))
        topology_dict = {"nodes": node_df, "edges": edge_df}

        fig = plt.figure(figsize=self.figsize)

        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        self._draw_topology(
            ax_main,
            node_df,
            edge_df,
            show_labels,
            show_grid=show_grid,
            axis_style="axes",
        )

        ax_degree = plt.subplot2grid((3, 3), (0, 2))
        self._plot_degree_distribution(ax_degree, node_df)

        ax_isd = plt.subplot2grid((3, 3), (1, 2))
        self._plot_isd_statistics(ax_isd, node_df, edge_df)

        ax_links = plt.subplot2grid((3, 3), (2, 2))
        self._plot_link_statistics(ax_links, edge_df)

        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        if write_extras:
            self._create_individual_plots(topology_dict, output_path.parent, dpi=dpi)
        
    def _draw_topology(
        self,
        ax,
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        show_labels: bool,
        show_grid: bool = True,
        *,
        axis_style: str = "axes",
        label_core_only: bool = False,
    ) -> None:
        """Draw the main topology map: all links, core vs non-core AS, legends.

        ``axis_style``:
            ``"axes"`` — coordinates, grid (dashboard main panel).
            ``"off"`` — minimal frame for standalone PNG export.
        """

        G = nx.Graph()
        for _, node in node_df.iterrows():
            G.add_node(int(node["as_id"]), **{k: v for k, v in node.items()})

        seen_edges: Set[Tuple[int, int]] = set()
        for _, edge in edge_df.iterrows():
            edge_key = tuple(sorted((int(edge["u"]), int(edge["v"]))))
            if edge_key not in seen_edges:
                G.add_edge(int(edge["u"]), int(edge["v"]), type=str(edge["type"]))
                seen_edges.add(edge_key)

        pos = {int(row["as_id"]): (float(row["x"]), float(row["y"])) for _, row in node_df.iterrows()}

        self._draw_isd_regions(ax, node_df, pos)

        # One pass per distinct edge type present (so PEER / unknown types still render)
        types_in_graph = sorted({d.get("type", "UNKNOWN") for _, _, d in G.edges(data=True)})
        for link_type in types_in_graph:
            style = self.LINK_STYLES.get(str(link_type), self.DEFAULT_LINK_STYLE)
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == link_type]
            if not edges:
                continue
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=style["color"],
                width=style["width"],
                style=style["style"],
                alpha=0.75,
                ax=ax,
            )

        core_nodes = node_df[node_df["role"] == "core"]["as_id"].astype(int).tolist()
        non_core_nodes = node_df[node_df["role"] == "non-core"]["as_id"].astype(int).tolist()

        if non_core_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=non_core_nodes,
                node_color=self.NON_CORE_COLOR,
                node_size=200,
                alpha=0.85,
                ax=ax,
            )
        if core_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=core_nodes,
                node_color=self.CORE_COLOR,
                node_size=500,
                node_shape="s",
                alpha=0.9,
                ax=ax,
            )

        if show_labels:
            if label_core_only:
                core_set = set(node_df[node_df["role"] == "core"]["as_id"].astype(int))
                labels = {n: str(n) for n in G.nodes() if int(n) in core_set}
            else:
                labels = {n: str(n) for n in G.nodes()}
            if labels:
                nx.draw_networkx_labels(
                    G,
                    pos,
                    labels,
                    font_size=7 if len(G) > 120 else 9,
                    font_color="white",
                    font_weight="bold",
                    ax=ax,
                )
            if label_core_only and len(G) > len(labels):
                ax.text(
                    0.02,
                    0.02,
                    f"{len(G)} ASes — only core AS IDs labeled",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
                )

        as_legends = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=self.CORE_COLOR,
                markersize=11,
                label="Core AS",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.NON_CORE_COLOR,
                markersize=9,
                label="Non-core AS (stub / child)",
            ),
        ]
        leg_as = ax.legend(handles=as_legends, title="AS role", loc="upper left", frameon=True)
        ax.add_artist(leg_as)

        link_legends = []
        for lt in types_in_graph:
            st = self.LINK_STYLES.get(str(lt), self.DEFAULT_LINK_STYLE)
            link_legends.append(
                Line2D(
                    [0],
                    [0],
                    color=st["color"],
                    linewidth=st["width"] / 2,
                    linestyle=st["style"],
                    label=f"{lt} link",
                )
            )
        ax.legend(handles=link_legends, title="Link type (all AS–AS links)", loc="lower left", frameon=True)

        ax.set_title("SCION topology (geographic layout)", fontsize=16, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.06)
        if axis_style == "off":
            ax.axis("off")
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel("X coordinate", fontsize=11)
            ax.set_ylabel("Y coordinate", fontsize=11)
            if show_grid:
                ax.grid(True, alpha=0.25)
            else:
                ax.grid(False)
        
    def _draw_isd_regions(self, ax, node_df: pd.DataFrame, pos: Dict):
        """Draw convex hulls around ISDs"""
        from scipy.spatial import ConvexHull
        
        for isd in sorted(node_df['isd'].unique()):
            isd_nodes = node_df[node_df['isd'] == isd]['as_id'].tolist()
            if len(isd_nodes) < 3:
                continue
                
            # Get positions
            points = np.array([pos[n] for n in isd_nodes])
            
            # Compute convex hull
            try:
                hull = ConvexHull(points)
                
                # Draw hull with transparency
                hull_points = points[hull.vertices]
                color = self.ISD_COLORS[isd % len(self.ISD_COLORS)]
                
                # Add padding
                center = hull_points.mean(axis=0)
                hull_points = center + 1.1 * (hull_points - center)
                
                patch = plt.Polygon(
                    hull_points,
                    alpha=0.2,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=2,
                    linestyle='--'
                )
                ax.add_patch(patch)
                
                # Add ISD label
                ax.text(center[0], center[1], f'ISD {isd}',
                       fontsize=20, fontweight='bold',
                       ha='center', va='center',
                       color=color, alpha=0.7)
                       
            except Exception:
                # Skip if hull computation fails
                pass
                
    def _plot_degree_distribution(self, ax, node_df: pd.DataFrame):
        """Plot degree distribution"""
        degrees = node_df["degree"].values
        if len(degrees) == 0:
            ax.set_title("Degree Distribution (empty)", fontsize=12)
            return
        dmin, dmax = int(degrees.min()), int(degrees.max())
        if dmin == dmax:
            bins = [dmin, dmin + 1]
        else:
            bins = range(dmin, dmax + 2)

        ax.hist(degrees, bins=bins, alpha=0.7, color="#3498DB", edgecolor="black")
        
        ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_degree = degrees.mean()
        ax.axvline(mean_degree, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_degree:.1f}')
        ax.legend()
        
    def _plot_isd_statistics(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        """Plot ISD composition"""
        if len(node_df) == 0:
            ax.set_title("ISD Composition (empty)", fontsize=12)
            return
        isd_stats = []

        for isd in sorted(node_df["isd"].unique()):
            isd_nodes = node_df[node_df['isd'] == isd]
            
            stats = {
                'ISD': isd,
                'Total': len(isd_nodes),
                'Core': len(isd_nodes[isd_nodes['role'] == 'core']),
                'Non-core': len(isd_nodes[isd_nodes['role'] == 'non-core'])
            }
            isd_stats.append(stats)
            
        isd_df = pd.DataFrame(isd_stats)
        
        # Stacked bar chart
        x = np.arange(len(isd_df))
        width = 0.6
        
        ax.bar(x, isd_df['Core'], width, label='Core',
               color=self.CORE_COLOR, alpha=0.8)
        ax.bar(x, isd_df['Non-core'], width, bottom=isd_df['Core'],
               label='Non-core', color=self.NON_CORE_COLOR, alpha=0.8)
        
        ax.set_title('ISD Composition', fontsize=12, fontweight='bold')
        ax.set_xlabel('ISD')
        ax.set_ylabel('Number of ASes')
        ax.set_xticks(x)
        ax.set_xticklabels(isd_df['ISD'])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
    def _plot_link_statistics(self, ax, edge_df: pd.DataFrame):
        """Plot link type distribution"""
        if len(edge_df) == 0:
            ax.set_title("Link Type Distribution (empty)", fontsize=12)
            return
        link_counts = edge_df["type"].value_counts()
        colors = [
            self.LINK_STYLES.get(str(lt), self.DEFAULT_LINK_STYLE)["color"]
            for lt in link_counts.index
        ]
        
        wedges, texts, autotexts = ax.pie(
            link_counts.values,
            labels=link_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Link Type Distribution', fontsize=12, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
    def _create_individual_plots(
        self, topology: Dict, output_dir: Path, *, dpi: int = 300
    ) -> None:
        """Create additional individual visualizations."""
        node_df = topology["nodes"]
        edge_df = topology["edges"]

        fig, ax = plt.subplots(figsize=(10, 8))
        self._create_isd_map(ax, node_df)
        plt.savefig(output_dir / "isd_map.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        self._create_core_network(ax, node_df, edge_df)
        plt.savefig(output_dir / "core_network.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        self._create_connectivity_matrix(ax, node_df, edge_df)
        plt.savefig(output_dir / "connectivity_matrix.png", dpi=dpi, bbox_inches="tight")
        plt.close()
        
    def _create_isd_map(self, ax, node_df: pd.DataFrame):
        """Create ISD membership map"""
        # Scatter plot colored by ISD
        for isd in sorted(node_df['isd'].unique()):
            isd_nodes = node_df[node_df['isd'] == isd]
            color = self.ISD_COLORS[isd % len(self.ISD_COLORS)]
            
            # Plot non-core nodes
            non_core = isd_nodes[isd_nodes['role'] == 'non-core']
            ax.scatter(non_core['x'], non_core['y'], 
                      c=color, s=100, alpha=0.6,
                      label=f'ISD {isd}')
            
            # Plot core nodes
            core = isd_nodes[isd_nodes['role'] == 'core']
            ax.scatter(core['x'], core['y'],
                      c=color, s=300, marker='s',
                      edgecolors='black', linewidths=2)
                      
        ax.set_title('ISD Membership Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _create_core_network(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        """Visualize only the core network"""
        core_nodes = node_df[node_df["role"] == "core"]["as_id"].astype(int).tolist()
        core_edges = edge_df[edge_df["type"].isin(("core", "CORE"))]
        
        # Create core graph
        G = nx.Graph()
        for node in core_nodes:
            node_data = node_df[node_df["as_id"] == node].iloc[0]
            G.add_node(int(node), isd=int(node_data["isd"]))

        for _, edge in core_edges.iterrows():
            u, v = int(edge["u"]), int(edge["v"])
            if u in core_nodes and v in core_nodes:
                G.add_edge(u, v)
                
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes colored by ISD
        for isd in sorted(node_df['isd'].unique()):
            isd_core_nodes = [n for n in G.nodes() 
                             if G.nodes[n]['isd'] == isd]
            if isd_core_nodes:
                color = self.ISD_COLORS[isd % len(self.ISD_COLORS)]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=isd_core_nodes,
                    node_color=color, node_size=1000,
                    node_shape='s', alpha=0.8, ax=ax
                )
                
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color=self.LINK_STYLES['core']['color'],
            width=3, alpha=0.7, ax=ax
        )
        
        # Labels
        nx.draw_networkx_labels(
            G, pos, font_size=12, font_color='white',
            font_weight='bold', ax=ax
        )
        
        ax.set_title('Core AS Network', fontsize=16, fontweight='bold')
        ax.axis('off')
        
    def _create_connectivity_matrix(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        """Create adjacency matrix visualization"""
        n_nodes = len(node_df)
        matrix = np.zeros((n_nodes, n_nodes))

        ordered = node_df.sort_values("as_id").reset_index(drop=True)
        as_to_idx = {int(row["as_id"]): i for i, row in ordered.iterrows()}
        
        # Fill matrix
        for _, edge in edge_df.iterrows():
            i = as_to_idx.get(edge['u'], -1)
            j = as_to_idx.get(edge['v'], -1)
            if i >= 0 and j >= 0:
                # Color code by link type
                et = str(edge["type"])
                link_value = {
                    "core": 4,
                    "CORE": 4,
                    "parent-child": 3,
                    "PARENT_CHILD": 3,
                    "child-parent": 2,
                    "CHILD_PARENT": 2,
                    "peer": 1,
                    "PEER": 1,
                }.get(et, 0)
                
                matrix[i, j] = link_value
                matrix[j, i] = link_value  # Symmetric
                
        # Plot
        im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(['None', 'Peer', 'Child-Parent', 'Parent-Child', 'Core'])
        
        ax.set_title('Connectivity Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('AS ID Index')
        ax.set_ylabel('AS ID Index')
        
        # ISD boundaries (matrix rows/cols follow ``ordered`` sort by AS id)
        isd_boundaries = []
        current_idx = 0
        for isd in sorted(ordered["isd"].unique()):
            isd_size = len(ordered[ordered["isd"] == isd])
            isd_boundaries.append(current_idx + isd_size)
            current_idx += isd_size

        for boundary in isd_boundaries[:-1]:
            ax.axhline(boundary - 0.5, color="black", linewidth=2)
            ax.axvline(boundary - 0.5, color="black", linewidth=2)


def render_scion_topology_png(
    topology_path: Union[str, Path],
    output_png: Union[str, Path],
    *,
    title: Optional[str] = None,
    show_labels: bool = True,
    dpi: int = 200,
    show_interactive: bool = False,
) -> Path:
    """Single-panel geographic map (pickle or JSON). Saves PNG; optional ``plt.show()``."""
    topology_path = Path(topology_path)
    output_png = Path(output_png)
    node_df, edge_df, _meta = load_topology_tables(topology_path)
    n = len(node_df)
    w = min(10.0 + 0.12 * min(n, 400), 28.0)
    h = min(w * 0.85, 24.0)

    tv = TopologyVisualizer(figsize=(min(int(w), 28), min(int(h), 24)))
    fig, ax = plt.subplots(figsize=(w, h))
    label_core_only = bool(show_labels and n > 350)
    tv._draw_topology(
        ax,
        node_df,
        edge_df,
        show_labels,
        show_grid=False,
        axis_style="off",
        label_core_only=label_core_only,
    )
    ttl = title or f"SCION topology ({topology_path.name})"
    ax.set_title(ttl, fontsize=14, pad=10)

    core_ct = len(node_df[node_df["role"] == "core"])
    stats = (
        f"ASes: {n}   Links (undirected): {len(edge_df)}   "
        f"Core: {core_ct}   Non-core: {n - core_ct}"
    )
    fig.subplots_adjust(bottom=0.09)
    fig.text(0.5, 0.02, stats, ha="center", fontsize=10, color="#333333")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show_interactive:
        plt.show()
    plt.close(fig)
    return output_png


def create_topology_report(topology_path: Path, output_dir: Path):
    """
    Create a comprehensive topology report with visualizations and statistics.

    ``topology_path`` may be a **pickle** topology or ``scion_topology.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_df, edge_df, _meta = load_topology_tables(Path(topology_path))
    topology = {"nodes": node_df, "edges": edge_df}

    visualizer = TopologyVisualizer()
    visualizer.visualize_topology(
        Path(topology_path),
        output_dir / "topology_overview.png",
    )

    stats_report = generate_topology_stats(topology)
    
    with open(output_dir / 'topology_stats.txt', 'w') as f:
        f.write(stats_report)
        
    print(f"Topology report generated in {output_dir}")
    

def generate_topology_stats(topology: Dict) -> str:
    """Generate detailed statistics report"""
    node_df = topology['nodes']
    edge_df = topology['edges']
    
    report = []
    report.append("=== SCION Topology Statistics Report ===\n")
    
    # Basic stats
    report.append(f"Total ASes: {len(node_df)}")
    report.append(f"Total Links: {len(edge_df)}")
    report.append(f"Number of ISDs: {len(node_df['isd'].unique())}")
    report.append(f"Core ASes: {len(node_df[node_df['role'] == 'core'])}")
    if len(node_df):
        report.append(f"Average Degree: {node_df['degree'].mean():.2f}")
    else:
        report.append("Average Degree: n/a")
    
    # ISD details
    report.append("\n=== ISD Breakdown ===")
    for isd in sorted(node_df['isd'].unique()):
        isd_nodes = node_df[node_df['isd'] == isd]
        n_core = len(isd_nodes[isd_nodes['role'] == 'core'])
        n_total = len(isd_nodes)
        report.append(f"ISD {isd}: {n_total} ASes ({n_core} core, {n_total-n_core} non-core)")
        
    # Link analysis
    report.append("\n=== Link Analysis ===")
    if len(edge_df) == 0:
        report.append("(no edges)")
    else:
        link_counts = edge_df["type"].value_counts()
        for link_type, count in link_counts.items():
            percentage = (count / len(edge_df)) * 100
            report.append(f"{link_type}: {count} ({percentage:.1f}%)")
        
    # Connectivity
    report.append("\n=== Connectivity Metrics ===")
    
    # Check if graph is connected
    G = nx.Graph()
    for _, edge in edge_df.iterrows():
        G.add_edge(edge['u'], edge['v'])
        
    if nx.is_connected(G):
        report.append("✓ Topology is fully connected")
        diameter = nx.diameter(G)
        report.append(f"Network diameter: {diameter}")
    else:
        components = list(nx.connected_components(G))
        report.append(f"⚠ Topology has {len(components)} connected components")
        
    return '\n'.join(report)