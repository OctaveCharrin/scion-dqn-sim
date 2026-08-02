from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from src.pipeline.intent_cond_eval import (  # noqa: E402
    INTENT_LABELS,
    INTENT_PROFILES,
    METHOD_LABELS,
)
from src.pipeline.intent_cond_figures import (  # noqa: E402
    CH6_METHOD_COLORS,
    CH6_METHOD_MARKERS,
    INTENT_COLORS,
    _METHOD_ORDER,
)
from src.pipeline.figures import (
    COLUMN_WIDTH,
    FULL_WIDTH,
    apply_lncs_style,
)  # noqa: E402,E501

# Per-seed traces are drawn thin and translucent so the mean stays the figure's
# subject and the spread reads as texture around it.
SEED_LINE = dict(linewidth=0.8, alpha=0.35, zorder=2)
SEED_DOT = dict(s=9, alpha=0.55, zorder=4, linewidths=0)
BAND_ALPHA = 0.18

_INTERP_COLORS = {
    "conditional_concat_2stream": "#009E73",
    "conditional_concat": "#E69F00",
}
_INTERP_MARKERS = {"conditional_concat_2stream": "^", "conditional_concat": "s"}
_PATHCOUNT_COLORS = {
    "flat_dqn": "#D55E00",
    "scoring_enhanced": "#999999",
    "conditional_concat": "#E69F00",
    "conditional_concat_2stream": "#009E73",
}
_PATHCOUNT_MARKERS = {
    "flat_dqn": "v",
    "scoring_enhanced": "X",
    "conditional_concat": "s",
    "conditional_concat_2stream": "^",
}


def _seed_note(n: int, kind: str = "Bands") -> str:
    """Uniform wording for the spread annotation every figure carries."""
    return f"{kind}: 95% CI over {n} training seed{'' if n == 1 else 's'}"


def _n_seeds(rows: Sequence[Dict[str, str]]) -> int:
    return int(_f(rows[0], "n_seeds")) if rows else 0


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def seedvals(row: Dict[str, str], col: str) -> np.ndarray:
    """The individual per-seed values ``analyze_seed_results`` kept beside the CI."""
    raw = row.get(f"{col}_seedvals", "")
    return np.asarray([float(v) for v in raw.split(";") if v], dtype=float)


def band(rows: Sequence[Dict[str, str]], col: str) -> tuple:
    """Confidence-band edges for a series of aggregate rows."""
    return (
        np.array([_f(r, f"{col}_ci_lo") for r in rows]),
        np.array([_f(r, f"{col}_ci_hi") for r in rows]),
    )


# ----------------------------------------------------------- 1. intent heatmap
def plot_intent_heatmap_seeds(
    matrix_csv: Path, claims_json: Path, out_path: Path
) -> Path:
    """R(intent told, objective scored), mean over seeds, with the alignment test.

    Colour is still each cell's deviation from its column mean on a symmetric
    diverging scale, but two things are added that the single-run figure could
    not carry. Each cell prints its 95% half-width under the mean, and each
    diagonal cell is boxed solid only when the margin over the best off-diagonal
    intent has an interval that clears zero -- a dashed box and an ``n.s.`` tag
    otherwise. The column range is printed in the axis label, because the
    column-relative colouring deliberately hides how different the columns are.
    """
    apply_lncs_style()
    rows = _read_csv(matrix_csv)
    claims = json.loads(claims_json.read_text())
    cols = claims["intent_matrix"]["columns"]
    n_seeds = claims.get("n_seeds", 5)

    intents = INTENT_PROFILES
    idx = {n: i for i, n in enumerate(intents)}
    mean = np.full((len(intents), len(intents)), np.nan)
    half = np.full_like(mean, np.nan)
    for r in rows:
        i, j = idx[r["intent_told"]], idx[r["intent_scored"]]
        mean[i, j] = _f(r, "reward_mean_mean")
        half[i, j] = (_f(r, "reward_mean_ci_hi") - _f(r, "reward_mean_ci_lo")) / 2.0

    dev = np.full_like(mean, np.nan)
    for j in range(mean.shape[1]):
        col = mean[:, j]
        dev[:, j] = col - float(col[np.isfinite(col)].mean())
    vmax = max(float(np.abs(dev[np.isfinite(dev)]).max()), 1e-6)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, COLUMN_WIDTH + 1.0))
    im = ax.imshow(dev, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    xlabels = [
        f"{INTENT_LABELS[n]}\n(range {cols[n]['column_range']['mean']:.3f})"
        for n in intents
    ]
    ax.set_xticks(range(len(intents)))
    ax.set_yticks(range(len(intents)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticklabels([INTENT_LABELS[n] for n in intents])
    ax.set_xlabel("Objective the choice is scored under")
    ax.set_ylabel("Intent the agent is conditioned on")

    for i in range(len(intents)):
        for j in range(len(intents)):
            if not np.isfinite(mean[i, j]):
                continue
            ink = "white" if abs(dev[i, j]) / vmax > 0.6 else "black"
            ax.text(
                j,
                i - 0.13,
                f"{mean[i, j]:.3f}",
                ha="center",
                va="center",
                color=ink,
                fontsize=8,
                fontweight="bold" if i == j else "normal",
            )
            ax.text(
                j,
                i + 0.17,
                f"±{half[i, j]:.3f}",
                ha="center",
                va="center",
                color=ink,
                fontsize=6.2,
            )
            if i != j:
                continue
            significant = cols[intents[j]]["margin_ci_excludes_zero"]
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#000000" if significant else "#555555",
                    linestyle="-" if significant else (0, (3, 2)),
                    lw=2.2 if significant else 1.4,
                )
            )
            if not significant:
                ax.text(
                    j + 0.36,
                    i - 0.36,
                    "n.s.",
                    ha="right",
                    va="center",
                    fontsize=6.5,
                    style="italic",
                    color=ink,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reward vs. column mean\n(per objective)")
    n_sig = sum(1 for c in cols.values() if c["margin_ci_excludes_zero"])
    ax.set_title(
        f"{_seed_note(n_seeds, 'Cells')} · solid box = diagonal margin significant "
        f"({n_sig}/{len(intents)} columns)",
        fontsize=6.8,
        color="#333333",
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------- 2. intent boxplots
def plot_intent_boxplots_seeds(
    boxstats_csv: Path, summary_csv: Path, out_path: Path
) -> Path:
    """Chosen-path metric distributions per intent, with each seed's mean marked.

    The box bodies pool every selection from every seed, which is the
    distribution described; the overlaid dots are the five per-seed
    means, so a reader sees at once whether the separation between intents is
    larger than the separation between training runs.
    """
    apply_lncs_style()
    stats = {(r["intent"], r["metric"]): r for r in _read_csv(boxstats_csv)}
    summary = {r["intent"]: r for r in _read_csv(summary_csv)}

    panels = [
        (
            "latency_ms",
            "latency_mean_ms",
            "Chosen latency (ms)",
            "delay_averse",
            "lower",
        ),
        (
            "bandwidth_mbps",
            "goodput_mean_mbps",
            "Chosen goodput (Mbps)",
            "bandwidth_max",
            "higher",
        ),
        ("trust", "trust_mean", "Chosen path trust", "delay_averse", "higher"),
        (
            None,
            "loss_exposure_pct",
            "Selections with loss > 0 (%)",
            "loss_averse",
            "lower",
        ),
    ]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(FULL_WIDTH + 0.6, 3.2), constrained_layout=True
    )
    labels = [INTENT_LABELS[n] for n in INTENT_PROFILES]

    for ax, (metric, summary_col, ylabel, target, better) in zip(axes, panels):
        positions = range(1, len(INTENT_PROFILES) + 1)
        if metric is None:
            # ~99% of chosen paths have exactly zero loss, so a boxplot is a flat
            # line at zero; the bar is the share of selections that see any loss.
            vals = [_f(summary[n], f"{summary_col}_mean") for n in INTENT_PROFILES]
            err = np.array(
                [
                    [
                        _f(summary[n], f"{summary_col}_mean")
                        - _f(summary[n], f"{summary_col}_ci_lo")
                        for n in INTENT_PROFILES
                    ],
                    [
                        _f(summary[n], f"{summary_col}_ci_hi")
                        - _f(summary[n], f"{summary_col}_mean")
                        for n in INTENT_PROFILES
                    ],
                ]
            )
            bars = ax.bar(positions, vals, width=0.6, zorder=3)
            for bar, name in zip(bars, INTENT_PROFILES):
                bar.set_facecolor(INTENT_COLORS[name])
                bar.set_alpha(0.85)
                is_target = name == target
                bar.set_edgecolor("#000000" if is_target else "#333333")
                bar.set_linewidth(2.2 if is_target else 1.0)
            ax.errorbar(
                list(positions),
                vals,
                yerr=np.clip(err, 0, None),
                fmt="none",
                ecolor="#222222",
                elinewidth=1.1,
                capsize=3,
                zorder=5,
            )
        else:
            bxp_stats = []
            for name in INTENT_PROFILES:
                s = stats[(name, metric)]
                bxp_stats.append(
                    {
                        "med": _f(s, "median"),
                        "q1": _f(s, "q1"),
                        "q3": _f(s, "q3"),
                        "whislo": _f(s, "p5"),
                        "whishi": _f(s, "p95"),
                        "fliers": [],
                        "label": INTENT_LABELS[name],
                    }
                )
            bp = ax.bxp(
                bxp_stats,
                positions=list(positions),
                patch_artist=True,
                showfliers=False,
                widths=0.6,
                zorder=2,
            )
            for patch, name in zip(bp["boxes"], INTENT_PROFILES):
                patch.set_facecolor(INTENT_COLORS[name])
                patch.set_alpha(0.85)
                is_target = name == target
                patch.set_edgecolor("#000000" if is_target else "#333333")
                patch.set_linewidth(2.2 if is_target else 1.0)
            for median in bp["medians"]:
                median.set_color("black")

        # Per-seed means:
        for pos, name in zip(positions, INTENT_PROFILES):
            vals = seedvals(summary[name], summary_col)
            ax.scatter(
                np.full(vals.size, pos),
                vals,
                color="#111111",
                marker="_",
                s=90,
                linewidths=1.3,
                zorder=6,
            )

        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(
            f"{INTENT_LABELS[target]} intent → {better}", fontsize=8, color="#333333"
        )
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].text(
        0.02,
        0.98,
        f"— = per-seed mean ({_n_seeds(list(summary.values()))} seeds)",
        transform=axes[0].transAxes,
        fontsize=6.5,
        va="top",
        color="#333333",
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- 3. zero-shot
def plot_zeroshot_seeds(interp_csv: Path, out_path: Path) -> Path:
    """Chosen latency and goodput along the unseen intent segment, per seed.

    Each seed's own curve is drawn thin behind the seed mean and its confidence
    band, so monotonicity -- which is a per-seed property, not a property of the
    average -- can be checked by eye on every run rather than inferred from the
    mean curve.
    """
    apply_lncs_style()
    rows = _read_csv(interp_csv)
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH + 0.6, 3.1), constrained_layout=True
    )
    for method in ("conditional_concat", "conditional_concat_2stream"):
        recs = sorted(by_method.get(method, []), key=lambda r: _f(r, "t"))
        if not recs:
            continue
        ts = np.array([_f(r, "t") for r in recs])
        color = _INTERP_COLORS[method]
        for ax, col, ylabel in (
            (axes[0], "latency_mean_ms", "Chosen latency (ms)"),
            (axes[1], "goodput_mean_mbps", "Chosen goodput (Mbps)"),
        ):
            per_seed = np.vstack([seedvals(r, col) for r in recs])  # (t, seed)
            for s in range(per_seed.shape[1]):
                ax.plot(ts, per_seed[:, s], color=color, **SEED_LINE)
            lo, hi = band(recs, col)
            ax.fill_between(
                ts, lo, hi, color=color, alpha=BAND_ALPHA, zorder=3, linewidth=0
            )
            ax.plot(
                ts,
                [_f(r, f"{col}_mean") for r in recs],
                marker=_INTERP_MARKERS[method],
                markersize=5,
                linewidth=1.9,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=5,
                label=METHOD_LABELS[method],
            )
            ax.set_ylabel(ylabel)

    for ax in axes:
        ax.axvspan(0.0, 1.0, color="#000000", alpha=0.04, zorder=0)
        for t in (0.0, 1.0):
            ax.axvline(t, color="#666666", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Interpolation coefficient $t$")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best", framealpha=0.9)
    axes[0].set_title(
        f"Thin lines: individual seeds · {_seed_note(_n_seeds(rows), 'band')}",
        fontsize=6.8,
        color="#333333",
        loc="left",
    )
    fig.suptitle(
        "$\\mathbf{w}(t) = (1-t)\\cdot$Throughput $+\\ t\\cdot$Low-Latency"
        "  — shaded band is interpolation; only $t=0,1$ were trained on",
        fontsize=7.5,
        color="#333333",
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- 4. path count
def plot_pathcount_seeds(pathcount_csv: Path, out_path: Path) -> Path:
    """Mean regret against the per-N best path, with every seed plotted.

    Regret spans four orders of magnitude across methods, so the axis is
    logarithmic and a symmetric confidence bar would run off the bottom for the
    scoring agents. Each seed is therefore drawn as its own faint marker beside
    the seed-mean line, which is the same information without the clipping.
    """
    apply_lncs_style()
    rows = _read_csv(pathcount_csv)
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.1, COLUMN_WIDTH + 0.1))
    order = [m for m in _PATHCOUNT_MARKERS if m in by_method]
    for method in order:
        recs = sorted(by_method[method], key=lambda r: _f(r, "n_paths"))
        ns = np.array([_f(r, "n_paths") for r in recs])
        color = _PATHCOUNT_COLORS[method]
        for i, r in enumerate(recs):
            vals = seedvals(r, "regret_mean")
            ax.scatter(
                np.full(vals.size, ns[i]) * np.linspace(0.94, 1.06, vals.size),
                np.clip(vals, 1e-6, None),
                color=color,
                marker=_PATHCOUNT_MARKERS[method],
                **SEED_DOT,
            )
        ax.plot(
            ns,
            [_f(r, "regret_mean_mean") for r in recs],
            marker=_PATHCOUNT_MARKERS[method],
            markersize=6,
            linewidth=1.9,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5,
            label=METHOD_LABELS[method],
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 4, 8, 16, 30])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Candidate paths $N$ (subsampled at decision time)")
    ax.set_ylabel("Mean regret vs best path in the set")
    ax.set_title(
        f"Line: mean over {_n_seeds(rows)} training seeds; faint markers: "
        "individual seeds",
        fontsize=6.8,
        color="#333333",
        loc="left",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------- 5. probing
def plot_probing_seeds(quality_csv: Path, out_path: Path) -> Path:
    """Goodput against probing cost per selection, with seed intervals.

    Only the learned selector carries an interval: the heuristics are
    deterministic functions of the environment and are identical in all five
    runs, which is stated on the figure so the missing bars are not read as an
    omission.
    """
    apply_lncs_style()
    rows = {r["method"]: r for r in _read_csv(quality_csv)}
    order = [m for m in _METHOD_ORDER if m in rows]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.1, COLUMN_WIDTH + 0.1))
    for method in order:
        r = rows[method]
        x = _f(r, "probe_cost_per_selection_ms_mean")
        y = _f(r, "goodput_mean_mbps_mean")
        xerr = np.array(
            [
                [x - _f(r, "probe_cost_per_selection_ms_ci_lo")],
                [_f(r, "probe_cost_per_selection_ms_ci_hi") - x],
            ]
        )
        yerr = np.array(
            [
                [y - _f(r, "goodput_mean_mbps_ci_lo")],
                [_f(r, "goodput_mean_mbps_ci_hi") - y],
            ]
        )
        color = CH6_METHOD_COLORS.get(method, "#333333")
        ax.errorbar(
            x,
            y,
            xerr=np.clip(xerr, 0, None),
            yerr=np.clip(yerr, 0, None),
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            s=110,
            color=color,
            marker=CH6_METHOD_MARKERS.get(method, "o"),
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Probing cost per selection (ms)")
    ax.set_ylabel("Achieved goodput (Mbps)")
    ax.margins(0.18)
    ax.grid(alpha=0.3, zorder=0)
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
    ax.set_title(
        f"{_seed_note(_n_seeds(list(rows.values())), 'Bars')}. The heuristics are "
        "deterministic and identical in every seed.",
        fontsize=6.5,
        color="#333333",
        loc="left",
    )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------- 6. ceiling
def plot_ceiling_seeds(ceiling_csv: Path, out_path: Path) -> Path:
    """Goodput and reward against realized congestion, with seed bands.

    The two panels are bridging argument: the goodput ceiling
    descends while the reward stays flat. Both are drawn with the across-seed
    band so the descent can be compared against the run-to-run spread rather
    than taken from one curve.
    """
    apply_lncs_style()
    rows = _read_csv(ceiling_csv)
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    order = [m for m in _METHOD_ORDER if m in by_method]

    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH + 0.6, COLUMN_WIDTH + 0.2), constrained_layout=True
    )
    for ax, col, ylabel in (
        (axes[0], "goodput_mean_mbps", "Achieved goodput (Mbps)"),
        (axes[1], "reward_mean", "Composite reward"),
    ):
        for method in order:
            recs = sorted(by_method[method], key=lambda r: _f(r, "congestion_mid_mean"))
            xs = np.array([_f(r, "congestion_mid_mean") for r in recs])
            color = CH6_METHOD_COLORS.get(method, "#333333")
            lo, hi = band(recs, col)
            ax.fill_between(
                xs, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2
            )
            ax.plot(
                xs,
                [_f(r, f"{col}_mean") for r in recs],
                marker=CH6_METHOD_MARKERS.get(method, "o"),
                markersize=6,
                linewidth=2,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=4,
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_xlabel("Realized congestion (mean path utilization)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    if order:
        ns = {int(_f(r, "n")) for r in by_method[order[0]]}
        note = f"{ns.pop()} decisions per bin per method" if len(ns) == 1 else ""
        axes[0].set_title(
            f"{note} · {_seed_note(_n_seeds(rows))}",
            fontsize=7.0,
            color="#333333",
            loc="left",
        )
    axes[0].legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------- orchestration
#: Aggregate CSV -> the filename the thesis includes. Keeping the thesis-side
#: names means ``\includegraphics`` needs no edit when these replace the
#: single-run versions.
THESIS_FIGURES = {
    "p1eval_intent_heatmap.png": "intent_reward_matrix_seeds.csv",
    "p1eval_intent_boxplots.png": "intent_selection_boxstats_seeds.csv",
    "p1eval_zeroshot.png": "intent_interpolation_seeds.csv",
    "p1eval_pathcount.png": "pathcount_scaling_seeds.csv",
    "p1eval_probing.png": "probing_quality_seeds.csv",
    "p1eval_ceiling.png": "ceiling_by_congestion_seeds.csv",
}


def generate_seed_figures(
    aggregate_dir: Path, fig_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Render all six seeded figures from ``aggregate_dir``."""
    fig_dir = fig_dir or (aggregate_dir / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    claims = aggregate_dir / "claims.json"
    out: Dict[str, str] = {}

    out["p1eval_intent_heatmap.png"] = str(
        plot_intent_heatmap_seeds(
            aggregate_dir / "intent_reward_matrix_seeds.csv",
            claims,
            fig_dir / "p1eval_intent_heatmap.png",
        )
    )
    out["p1eval_intent_boxplots.png"] = str(
        plot_intent_boxplots_seeds(
            aggregate_dir / "intent_selection_boxstats_seeds.csv",
            aggregate_dir / "intent_selection_summary_seeds.csv",
            fig_dir / "p1eval_intent_boxplots.png",
        )
    )
    out["p1eval_zeroshot.png"] = str(
        plot_zeroshot_seeds(
            aggregate_dir / "intent_interpolation_seeds.csv",
            fig_dir / "p1eval_zeroshot.png",
        )
    )
    out["p1eval_pathcount.png"] = str(
        plot_pathcount_seeds(
            aggregate_dir / "pathcount_scaling_seeds.csv",
            fig_dir / "p1eval_pathcount.png",
        )
    )
    out["p1eval_probing.png"] = str(
        plot_probing_seeds(
            aggregate_dir / "probing_quality_seeds.csv",
            fig_dir / "p1eval_probing.png",
        )
    )
    out["p1eval_ceiling.png"] = str(
        plot_ceiling_seeds(
            aggregate_dir / "ceiling_by_congestion_seeds.csv",
            fig_dir / "p1eval_ceiling.png",
        )
    )
    return out
