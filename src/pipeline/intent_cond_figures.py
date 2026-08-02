from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.pipeline.intent_cond_eval import (
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
    # Whichever conditional variant is profiled is the hero of these figures, so
    # all three share the hero colour/marker; only one appears in a given CSV.
    "conditional_film": "#0072B2",
    "conditional_concat": "#0072B2",
    "conditional_concat_2stream": "#0072B2",
    "shortest_path": "#E69F00",
    "widest_path": "#009E73",
    "lowest_latency": "#D55E00",
    "ecmp": "#56B4E9",
    "scion_default": "#999999",
    "random": "#CC79A7",
}
CH6_METHOD_MARKERS: Dict[str, str] = {
    "conditional_film": "o",
    "conditional_concat": "o",
    "conditional_concat_2stream": "o",
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
    "conditional_concat",
    "conditional_concat_2stream",
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
    # The trust panel previously annotated "Low-Loss intent -> higher" while the
    # data show Low-Latency winning: canonical trust weights loss and delay
    # equally, so a delay-averse intent maximizes it. The panel now plots the
    # chosen path's *loss rate*, which is what the Low-Loss intent actually
    # optimizes, and trust keeps an honest annotation. Falls back to the old
    # three-panel layout when the CSV predates the extra columns.
    panels = [
        ("latency_ms", "Chosen latency (ms)", "delay_averse", "lower"),
        ("bandwidth_mbps", "Chosen goodput (Mbps)", "bandwidth_max", "higher"),
        ("trust", "Chosen path trust", "delay_averse", "higher"),
    ]
    # A loss boxplot is empty in this environment -- ~99% of chosen paths have
    # exactly zero loss -- so the Low-Loss panel shows the fraction of selections
    # that incur *any* loss instead, which is the quantity that actually varies.
    show_loss_bar = bool(rows) and "loss_rate" in rows[0]
    if show_loss_bar:
        panels.append(("loss_rate", "Selections with loss > 0 (%)", "loss_averse", "lower"))
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

    fig, axes = plt.subplots(
        1, len(panels), figsize=(FULL_WIDTH + 0.4, 3.0), constrained_layout=True
    )
    labels = [INTENT_LABELS[n] for n in intents]
    for ax, (col, ylabel, target, better) in zip(axes, panels):
        series = [data[col][name] for name in intents]
        if col == "loss_rate":
            # Percentage of selections incurring any loss at all.
            pct = [
                100.0 * float(np.mean(np.asarray(s) > 0.0)) if len(s) else 0.0
                for s in series
            ]
            bars = ax.bar(range(1, len(intents) + 1), pct, width=0.6)
            for bar, name, value in zip(bars, intents, pct):
                bar.set_facecolor(INTENT_COLORS[name])
                bar.set_alpha(0.85)
                is_target = name == target
                bar.set_edgecolor("#000000" if is_target else "#333333")
                bar.set_linewidth(2.2 if is_target else 1.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.set_ylim(0, max(pct) * 1.35 if max(pct) > 0 else 1.0)
        else:
            bp = ax.boxplot(series, patch_artist=True, showfliers=False, widths=0.6)
            for patch, name in zip(bp["boxes"], intents):
                patch.set_facecolor(INTENT_COLORS[name])
                patch.set_alpha(0.85)
                # Bold black edge on the intent this metric is the objective of,
                # so the eye lands on the box conditioning is expected to move.
                is_target = name == target
                patch.set_edgecolor("#000000" if is_target else "#333333")
                patch.set_linewidth(2.2 if is_target else 1.0)
            for median in bp["medians"]:
                median.set_color("black")
        ax.set_ylabel(ylabel, fontsize=8)
        target_label = INTENT_LABELS[target]
        ax.set_title(f"{target_label} intent → {better}", fontsize=8, color="#333333")
        ax.set_xticks(range(1, len(intents) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ------------------------------------------------ Figure 6.2 quality vs probe cost
def plot_quality_vs_probecost(
    quality_csv: Path, out_path: Path, *, metric: str = "goodput"
) -> Path:
    """Scatter: selection quality (y) vs probing cost per selection (x).

    The learned selector occupies the low-cost corner.
    """
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
    """Application performance vs realized congestion, per method.

    ``metric="both"`` renders the two-panel version the ceiling argument needs:
    goodput descending while reward stays flat is the clearest statement that the
    policy keeps doing its job and the *architecture* is the limit -- the reward
    normalizes goodput by the best candidate path at the same hour, so it is
    blind to the ceiling by construction.
    """
    apply_lncs_style()
    rows = _read_csv(ceiling_csv)
    metrics = ["goodput", "reward"] if metric == "both" else [metric]

    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    order = [m for m in _METHOD_ORDER if m in by_method]

    if len(metrics) == 1:
        fig, axes = plt.subplots(figsize=(COLUMN_WIDTH + 0.8, COLUMN_WIDTH - 0.1))
        axes = [axes]
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(FULL_WIDTH + 0.6, COLUMN_WIDTH + 0.1), constrained_layout=True
        )
        axes = list(axes)

    for ax, met in zip(axes, metrics):
        ycol = _METRIC_COLUMN[met]
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
        ax.set_ylabel(_METRIC_LABELS[met])
        ax.grid(alpha=0.3)
    # Bin-count annotation: every bin holds the same number of decisions per
    # method, so one number describes the whole figure.
    if order:
        ns = {int(_fnum(r, "n")) for r in by_method[order[0]]}
        if len(ns) == 1:
            axes[0].set_title(
                f"{ns.pop()} decisions per bin per method",
                fontsize=7.5,
                color="#333333",
                loc="left",
            )
    axes[0].legend(fontsize=7.5, loc="best", framealpha=0.9)
    if len(metrics) == 1:
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ------------------------------------------------- probing cost, broken out by intent
def plot_probing_by_intent(quality_csv: Path, out_path: Path) -> Path:
    """One panel per intent: goodput vs probes per selection.

    Supports the stronger claim that a single conditioned policy tracks the
    *per-intent strongest* heuristic (widest-path under Throughput,
    lowest-latency under Low-Latency) rather than only matching one heuristic
    under a Balanced objective.
    """
    apply_lncs_style()
    rows = _read_csv(quality_csv)
    profiles = []
    for r in rows:
        if r["profile"] not in profiles:
            profiles.append(r["profile"])

    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(FULL_WIDTH, 2.7), squeeze=False)
    for ax, profile in zip(axes[0], profiles):
        recs = [r for r in rows if r["profile"] == profile]
        for r in recs:
            method = r["method"]
            ax.scatter(
                _fnum(r, "probes_per_selection"),
                _fnum(r, "reward_mean"),
                s=70,
                marker=CH6_METHOD_MARKERS.get(method, "o"),
                color=CH6_METHOD_COLORS.get(method, "#333333"),
                edgecolor="white",
                linewidth=0.6,
                label=METHOD_LABELS.get(method, method),
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Probes per selection")
        ax.set_ylabel("Mean intent-weighted reward")
        ax.set_title(INTENT_LABELS.get(profile, profile), fontsize=8.5)
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        ncol=min(7, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- training curves
def plot_training_curves(
    stats_paths: Dict[str, Path], out_path: Path, *, smooth: int = 25
) -> Path:
    """Episode reward and training loss per agent, from the saved statistics files.

    Substantiates the convergence discussion, and shows directly whether the
    weakest rung of the ablation ladder is undertrained rather than
    architecturally limited.
    """
    import json

    apply_lncs_style()

    def _smooth(v: np.ndarray, k: int) -> np.ndarray:
        if k <= 1 or v.size < k:
            return v
        return np.convolve(v, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH + 0.6, 2.9), constrained_layout=True
    )
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    for i, (label, path) in enumerate(sorted(stats_paths.items())):
        if not Path(path).is_file():
            continue
        with open(path) as f:
            stats = json.load(f)
        color = palette[i % len(palette)]
        rewards = np.asarray(stats.get("episode_rewards", []), dtype=float)
        if rewards.size:
            sm = _smooth(rewards, smooth)
            axes[0].plot(
                np.arange(sm.size) + smooth // 2, sm, color=color, linewidth=1.8, label=label
            )
        losses = np.asarray(stats.get("losses", []), dtype=float)
        if losses.size:
            sm = _smooth(losses, max(smooth, 50))
            axes[1].plot(
                np.linspace(0, rewards.size if rewards.size else sm.size, sm.size),
                sm,
                color=color,
                linewidth=1.8,
                label=label,
            )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Episode reward ({smooth}-ep mov. avg.)", fontsize=8)
    axes[0].axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Training loss (moving average)", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_per_profile_training_curves(
    stats_path: Path, out_path: Path, *, smooth: int = 15
) -> Path:
    """Per-intent learning curves for a conditional agent.

    The stratified schedule assigns each episode one profile, so the saved
    ``episode_weight_profiles`` splits the run into one curve per intent -- the
    only place the two probe-frugality profiles appear at all.
    """
    import json

    apply_lncs_style()
    with open(stats_path) as f:
        stats = json.load(f)
    rewards = np.asarray(stats.get("episode_rewards", []), dtype=float)
    names = stats.get("episode_weight_profiles") or []
    if not names or rewards.size != len(names):
        raise ValueError(f"{stats_path} has no usable per-episode profile labels.")

    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.2, COLUMN_WIDTH - 0.1))
    for i, profile in enumerate(sorted(set(names))):
        idx = np.array([j for j, n in enumerate(names) if n == profile])
        vals = rewards[idx]
        if vals.size >= smooth:
            sm = np.convolve(vals, np.ones(smooth) / smooth, mode="valid")
            xs = idx[smooth // 2:][: sm.size]
        else:
            sm, xs = vals, idx
        ax.plot(
            xs,
            sm,
            color=palette[i % len(palette)],
            linewidth=1.7,
            label=INTENT_LABELS.get(profile, profile),
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode reward ({smooth}-ep moving average, per intent)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------ zero-shot / path-count
def plot_intent_interpolation(csv_path: Path, out_path: Path) -> Path:
    """Chosen latency and goodput against the intent-interpolation coefficient."""
    apply_lncs_style()
    rows = _read_csv(csv_path)
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    palette = {
        "conditional_film": "#0072B2",
        "conditional_concat": "#E69F00",
        "conditional_concat_2stream": "#009E73",
    }
    markers = {
        "conditional_film": "o",
        "conditional_concat": "s",
        "conditional_concat_2stream": "^",
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH + 0.6, 2.9), constrained_layout=True
    )
    for method, recs in by_method.items():
        recs = sorted(recs, key=lambda r: _fnum(r, "t"))
        ts = [_fnum(r, "t") for r in recs]
        for ax, col, ylabel in (
            (axes[0], "latency_mean_ms", "Chosen latency (ms)"),
            (axes[1], "goodput_mean_mbps", "Chosen goodput (Mbps)"),
        ):
            ax.plot(
                ts,
                [_fnum(r, col) for r in recs],
                marker=markers.get(method, "o"),
                markersize=5,
                linewidth=1.8,
                color=palette.get(method, "#333333"),
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=METHOD_LABELS.get(method, method),
            )
    for ax in axes:
        # Only t=0 and t=1 are weight vectors any agent has been trained on.
        ax.axvspan(0.0, 1.0, color="#000000", alpha=0.04, zorder=0)
        for t in (0.0, 1.0):
            ax.axvline(t, color="#666666", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Interpolation coefficient $t$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Chosen latency (ms)")
    axes[1].set_ylabel("Chosen goodput (Mbps)")
    axes[0].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle(
        "$\\mathbf{w}(t) = (1-t)\\cdot$Throughput $+\\ t\\cdot$Low-Latency   "
        "— shaded band is interpolation; only $t=0,1$ were trained on",
        fontsize=7.5,
        color="#333333",
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_pathcount_scaling(csv_path: Path, out_path: Path) -> Path:
    """Mean regret against the per-N best path, by candidate-set size."""
    apply_lncs_style()
    rows = _read_csv(csv_path)
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    palette = {
        "flat_dqn": "#D55E00",
        "scoring_enhanced": "#999999",
        "conditional_concat": "#E69F00",
        "conditional_concat_2stream": "#009E73",
        "conditional_film": "#0072B2",
    }
    markers = {
        "flat_dqn": "v",
        "scoring_enhanced": "X",
        "conditional_concat": "s",
        "conditional_concat_2stream": "^",
        "conditional_film": "o",
    }
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.8, COLUMN_WIDTH - 0.1))
    for method, recs in by_method.items():
        recs = sorted(recs, key=lambda r: _fnum(r, "n_paths"))
        ax.plot(
            [_fnum(r, "n_paths") for r in recs],
            [_fnum(r, "regret_mean") for r in recs],
            marker=markers.get(method, "o"),
            markersize=6,
            linewidth=1.8,
            color=palette.get(method, "#333333"),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Candidate paths $N$ (subsampled at decision time)")
    ax.set_ylabel("Mean regret vs best path in the set")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- orchestration
def generate_all_figures(
    artifact_dir: Path, fig_dir: Optional[Path] = None, *, metric: str = "goodput"
) -> Dict[str, str]:
    """Render all figures from the CSVs in ``artifact_dir``."""
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
    # Only the ceiling figure supports the two-panel "both" mode; the scatter
    # keeps a single quality axis.
    scatter_metric = "goodput" if metric == "both" else metric
    if quality_csv.is_file():
        outputs["fig_6_2_quality_vs_probe"] = str(
            plot_quality_vs_probecost(
                quality_csv,
                fig_dir / "fig_6_2_quality_vs_probe.png",
                metric=scatter_metric,
            )
        )
    if ceiling_csv.is_file():
        outputs["fig_6_3_ceiling"] = str(
            plot_ceiling_vs_congestion(
                ceiling_csv, fig_dir / "fig_6_3_ceiling.png", metric=metric
            )
        )
    by_intent_csv = artifact_dir / "probing_quality_by_intent.csv"
    if by_intent_csv.is_file():
        outputs["fig_probing_by_intent"] = str(
            plot_probing_by_intent(by_intent_csv, fig_dir / "fig_probing_by_intent.png")
        )
    interp_csv = artifact_dir / "intent_interpolation.csv"
    if interp_csv.is_file():
        outputs["fig_intent_interpolation"] = str(
            plot_intent_interpolation(
                interp_csv, fig_dir / "fig_intent_interpolation.png"
            )
        )
    pathcount_csv = artifact_dir / "pathcount_scaling.csv"
    if pathcount_csv.is_file():
        outputs["fig_pathcount_scaling"] = str(
            plot_pathcount_scaling(pathcount_csv, fig_dir / "fig_pathcount_scaling.png")
        )
    return outputs
