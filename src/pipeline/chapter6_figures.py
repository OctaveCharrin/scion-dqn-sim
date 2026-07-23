"""Publication figures for Chapter 6 (intent conditioning + single-path ceiling).

Reads the CSVs produced by ``src.pipeline.chapter6_eval`` and renders Figures
6.1 (intent alignment heatmap + selection-metric boxplots), 6.2 (selection
quality vs probing cost), and 6.3 (application performance vs congestion, the
single-path ceiling).

Colours use the colorblind-safe Okabe-Ito categorical set (validated with the
dataviz palette checker) for methods/intents and a CVD-safe sequential map
(``cividis``) for the reward heatmap. Marker shape is a redundant (non-colour)
encoding of method so the figures survive greyscale printing. Styling reuses the
LNCS look from ``src.pipeline.figures``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.pipeline.chapter6_eval import (
    INTENT_LABELS,
    INTENT_PROFILES,
    METHOD_LABELS,
)  # noqa: E402
from src.pipeline.figures import (
    COLUMN_WIDTH,
    FULL_WIDTH,
    apply_lncs_style,
)  # noqa: E402

# Colorblind-safe Okabe-Ito assignments (fixed order, never cycled).
CH6_METHOD_COLORS: Dict[str, str] = {
    "conditional_film": "#0072B2",
    "shortest_path": "#E69F00",
    "widest_path": "#009E73",
    "lowest_latency": "#D55E00",
    "ecmp": "#56B4E9",
    "scion_default": "#999999",
    "random": "#CC79A7",
}
CH6_METHOD_MARKERS: Dict[str, str] = {
    "conditional_film": "o",
    "shortest_path": "s",
    "widest_path": "^",
    "lowest_latency": "D",
    "ecmp": "P",
    "scion_default": "X",
    "random": "v",
}
INTENT_COLORS: Dict[str, str] = {
    "bandwidth_max": "#E69F00",
    "delay_averse": "#009E73",
    "loss_averse": "#0072B2",
    "balanced_extreme": "#CC79A7",
}

# Method draw order for legends (hero first).
_METHOD_ORDER = [
    "conditional_film",
    "shortest_path",
    "widest_path",
    "lowest_latency",
    "ecmp",
    "scion_default",
    "random",
]

_METRIC_LABELS = {
    "goodput": "Achieved goodput (Mbps)",
    "reward": "Composite reward",
}
_METRIC_COLUMN = {
    "goodput": "goodput_mean_mbps",
    "reward": "reward_mean",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fnum(row: Dict[str, str], key: str) -> float:
    val = row.get(key, "")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------ Figure 6.1 heatmap
def plot_intent_reward_heatmap(matrix_csv: Path, out_path: Path) -> Path:
    """R(intent_told, intent_scored) heatmap.

    Each cell is the mean reward when the agent is *conditioned on* the row intent
    and its chosen path is *scored under* the column objective. The alignment claim
    is column-wise: for a given objective, the matching (diagonal) intent should
    score highest. Objectives sit at very different reward levels, so we colour each
    cell by its **deviation from that column's mean** on a single shared, symmetric
    diverging scale. This keeps colour intensity proportional to the *actual* reward
    gap (a 0.005 spread stays near-neutral instead of being stretched to full
    range), so the colours match the annotated raw values. Red = the conditioning
    beats the average intent for that objective; blue = it does worse."""
    apply_lncs_style()
    rows = _read_csv(matrix_csv)
    intents = INTENT_PROFILES
    idx = {name: i for i, name in enumerate(intents)}
    mat = np.full((len(intents), len(intents)), np.nan)
    for r in rows:
        i = idx.get(r["intent_told"])
        j = idx.get(r["intent_scored"])
        if i is not None and j is not None:
            mat[i, j] = _fnum(r, "reward_mean")

    # Deviation of each cell from its column (objective) mean.
    dev = np.full_like(mat, np.nan)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            dev[:, j] = col - float(finite.mean())
    finite_dev = dev[np.isfinite(dev)]
    vmax = float(np.abs(finite_dev).max()) if finite_dev.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.6, COLUMN_WIDTH + 0.2))
    im = ax.imshow(dev, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    labels = [INTENT_LABELS[n] for n in intents]
    ax.set_xticks(range(len(intents)))
    ax.set_yticks(range(len(intents)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Objective the choice is scored under")
    ax.set_ylabel("Intent the agent is conditioned on")

    # Annotate every cell with the raw reward; outline the diagonal (matched intent).
    for i in range(len(intents)):
        for j in range(len(intents)):
            if not np.isfinite(mat[i, j]):
                continue
            txt_color = "white" if abs(dev[i, j]) / vmax > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8,
                fontweight="bold" if i == j else "normal",
            )
            if i == j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#000000",
                        lw=2.2,
                    )
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reward vs. column mean\n(per objective)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------- Figure 6.1 boxplots
def plot_intent_selection_boxplots(metrics_csv: Path, out_path: Path) -> Path:
    """Chosen-path latency / bandwidth / trust distributions per conditioning intent."""
    apply_lncs_style()
    rows = _read_csv(metrics_csv)
    intents = INTENT_PROFILES
    # Each panel is a metric; ``target`` is the intent that metric is the objective
    # of, and ``better`` is the direction that intent should push the chosen path.
    panels = [
        ("latency_ms", "Chosen latency (ms)", "delay_averse", "lower"),
        ("bandwidth_mbps", "Chosen goodput (Mbps)", "bandwidth_max", "higher"),
        ("trust", "Chosen path trust", "loss_averse", "higher"),
    ]
    data: Dict[str, Dict[str, List[float]]] = {
        col: {name: [] for name in intents} for col, _, _, _ in panels
    }
    for r in rows:
        intent = r.get("intent")
        if intent not in data[panels[0][0]]:
            continue
        for col, _, _, _ in panels:
            v = _fnum(r, col)
            if np.isfinite(v):
                data[col][intent].append(v)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9))
    labels = [INTENT_LABELS[n] for n in intents]
    for ax, (col, ylabel, target, better) in zip(axes, panels):
        series = [data[col][name] for name in intents]
        bp = ax.boxplot(series, patch_artist=True, showfliers=False, widths=0.6)
        for patch, name in zip(bp["boxes"], intents):
            patch.set_facecolor(INTENT_COLORS[name])
            patch.set_alpha(0.85)
            # Bold black edge on the intent this metric is the objective of, so the
            # eye lands on the box that conditioning is expected to move.
            is_target = name == target
            patch.set_edgecolor("#000000" if is_target else "#333333")
            patch.set_linewidth(2.2 if is_target else 1.0)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_ylabel(ylabel)
        target_label = INTENT_LABELS[target]
        ax.set_title(f"{target_label} intent → {better}", fontsize=8, color="#333333")
        ax.set_xticks(range(1, len(intents) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------ Figure 6.2 quality vs probe cost
def plot_quality_vs_probecost(
    quality_csv: Path, out_path: Path, *, metric: str = "goodput"
) -> Path:
    """Scatter: selection quality (y) vs probing cost per selection (x). FiLM = low-cost corner."""
    apply_lncs_style()
    rows = _read_csv(quality_csv)
    ycol = _METRIC_COLUMN[metric]
    ylabel = _METRIC_LABELS[metric]

    by_method = {r["method"]: r for r in rows}
    order = [m for m in _METHOD_ORDER if m in by_method]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.6, COLUMN_WIDTH))
    for method in order:
        r = by_method[method]
        x = _fnum(r, "probe_cost_per_selection_ms")
        y = _fnum(r, ycol)
        ax.scatter(
            x,
            y,
            s=110,
            color=CH6_METHOD_COLORS.get(method, "#333333"),
            marker=CH6_METHOD_MARKERS.get(method, "o"),
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Probing cost per selection (ms)")
    ax.set_ylabel(ylabel)
    ax.margins(0.16)
    ax.grid(alpha=0.3, zorder=0)
    # "better" points to the top-left corner (low probing cost, high quality).
    ax.annotate(
        "better",
        xy=(0.02, 0.98),
        xytext=(0.20, 0.86),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=8,
        ha="left",
        va="center",
        color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
    )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------ Figure 6.3 ceiling vs congestion
def plot_ceiling_vs_congestion(
    ceiling_csv: Path, out_path: Path, *, metric: str = "goodput"
) -> Path:
    """Line plot: application performance (y) vs realized congestion (x), per method."""
    apply_lncs_style()
    rows = _read_csv(ceiling_csv)
    ycol = _METRIC_COLUMN[metric]
    ylabel = _METRIC_LABELS[metric]

    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    order = [m for m in _METHOD_ORDER if m in by_method]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.8, COLUMN_WIDTH - 0.1))
    for method in order:
        recs = sorted(by_method[method], key=lambda r: _fnum(r, "congestion_mid"))
        xs = [_fnum(r, "congestion_mid") for r in recs]
        ys = [_fnum(r, ycol) for r in recs]
        ax.plot(
            xs,
            ys,
            marker=CH6_METHOD_MARKERS.get(method, "o"),
            markersize=6,
            linewidth=2,
            color=CH6_METHOD_COLORS.get(method, "#333333"),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Realized congestion (mean path utilization)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- orchestration
def generate_all_figures(
    artifact_dir: Path, fig_dir: Optional[Path] = None, *, metric: str = "goodput"
) -> Dict[str, str]:
    """Render all Chapter 6 figures from the CSVs in ``artifact_dir``."""
    fig_dir = fig_dir or (artifact_dir / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    matrix_csv = artifact_dir / "intent_reward_matrix.csv"
    metrics_csv = artifact_dir / "intent_selection_metrics.csv"
    quality_csv = artifact_dir / "probing_quality.csv"
    ceiling_csv = artifact_dir / "ceiling_by_congestion.csv"

    if matrix_csv.is_file():
        outputs["fig_6_1_heatmap"] = str(
            plot_intent_reward_heatmap(matrix_csv, fig_dir / "fig_6_1_heatmap.png")
        )
    if metrics_csv.is_file():
        outputs["fig_6_1_boxplots"] = str(
            plot_intent_selection_boxplots(
                metrics_csv, fig_dir / "fig_6_1_boxplots.png"
            )
        )
    if quality_csv.is_file():
        outputs["fig_6_2_quality_vs_probe"] = str(
            plot_quality_vs_probecost(
                quality_csv, fig_dir / "fig_6_2_quality_vs_probe.png", metric=metric
            )
        )
    if ceiling_csv.is_file():
        outputs["fig_6_3_ceiling"] = str(
            plot_ceiling_vs_congestion(
                ceiling_csv, fig_dir / "fig_6_3_ceiling.png", metric=metric
            )
        )
    return outputs
