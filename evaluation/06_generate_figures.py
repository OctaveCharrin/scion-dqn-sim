#!/usr/bin/env python3
"""
Generate LNCS-style figures for the evaluation results
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.figures import (
    COLUMN_WIDTH,
    FULL_WIDTH,
    METHOD_COLORS as method_colors,
    METHOD_DISPLAY_NAMES as method_display_names,
    apply_lncs_style,
    generate_figure2_for_profiles,
    plot_path_reward_boxplot,
)
from src.pipeline.run_dirs import resolve_run_dir

LABEL_ROTATION = 45  
LABEL_HA = "right"   

run_dir = resolve_run_dir()

with open(os.path.join(run_dir, "evaluation_results.json"), "r") as f:
    results = json.load(f)

summary = results["summary"]

apply_lncs_style()

# ---------------------------------------------------------------------------
# Figure 1: Probe Overhead and Selection Time (1x2 Subplots)
# ---------------------------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 4.0), constrained_layout=True)

# Sort methods by probe overhead
methods = list(summary.keys())
methods.sort(key=lambda m: summary[m]["avg_probe_time_per_selection"])

probe_times = [summary[m]["avg_probe_time_per_selection"] for m in methods]
sel_times = [summary[m]["avg_selection_time_ms"] for m in methods]
colors1 = [method_colors.get(m, "blue") for m in methods]
display_names1 = [method_display_names.get(m, m) for m in methods]

x_pos = np.arange(len(methods))

# Subplot 1: Probe overhead 
bars1 = ax1.bar(x_pos, probe_times, color=colors1, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Avg Probe Overhead (ms)")
ax1.set_title("Probe Overhead per Selection")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(display_names1, rotation=LABEL_ROTATION, ha=LABEL_HA)
# Add numbers on top of bars, pad Y-axis by 20% so text fits
ax1.bar_label(bars1, fmt='%.1f', padding=3, fontsize=8)
ax1.set_ylim(0, max(probe_times) * 1.2)

# Subplot 2: Selection Time
bars2 = ax2.bar(x_pos, sel_times, color=colors1, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Avg Selection Time (ms)")
ax2.set_title("Algorithm Selection Time")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(display_names1, rotation=LABEL_ROTATION, ha=LABEL_HA)
# Add numbers on top of bars, pad Y-axis by 20% so text fits
ax2.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)
ax2.set_ylim(0, max(sel_times) * 1.2)

fig1.savefig(os.path.join(run_dir, "figure1_probe_overhead.png"), dpi=300, bbox_inches="tight")
plt.close(fig1)

# ---------------------------------------------------------------------------
# Figure 2: Path Selection Reward (one plot per reward profile when available)
# ---------------------------------------------------------------------------
methods_by_reward = sorted(summary.keys(), key=lambda m: summary[m]["reward_mean"], reverse=True)

_multi_reward_path = os.path.join(run_dir, "multi_reward_comparison.json")
if os.path.isfile(_multi_reward_path):
    with open(_multi_reward_path, "r") as f:
        multi_reward = json.load(f)
    figure2_paths = generate_figure2_for_profiles(run_dir, multi_reward)
    for path in figure2_paths:
        print(f"Saved {path.name}")
else:
    plot_path_reward_boxplot(
        summary,
        os.path.join(run_dir, "figure2_path_reward.png"),
        title="Path Selection Performance (balanced)",
    )
    print("Saved figure2_path_reward.png (from evaluation_results; run eval_multi_reward_comparison for per-profile figures)")

# Generate comparison table
print("\n" + "="*80)
print("PERFORMANCE COMPARISON TABLE")
print("="*80)
print(f"{'Method':<20} {'Reward':<15} {'Latency (ms)':<15} {'Probes/Selection':<20} {'Reduction':<10}")
print("-"*80)

def _per_selection(method_summary):
    n = max(1, int(method_summary.get('n_selections', 336)))
    return method_summary['total_probes'] / n


_DQN_LIKE = (
    "dqn",
    "simple_dqn",
    "scoring_simple_dqn",
    "scoring_enhanced_dqn",
    "conditional_dqn",
    "scoring_dqn",
)

baseline_probes = np.mean(
    [_per_selection(summary[m]) for m in summary if m not in _DQN_LIKE]
)

for method in methods_by_reward:
    reward = f"{summary[method]['reward_mean']:.3f} ± {summary[method]['reward_std']:.3f}"
    latency = f"{summary[method]['latency_mean']:.1f}"
    probes = _per_selection(summary[method])

    if method in _DQN_LIKE:
        reduction = (
            f"{(baseline_probes - probes) / baseline_probes * 100:.1f}%"
            if baseline_probes
            else "-"
        )
    else:
        reduction = "-"

    print(
        f"{method_display_names[method]:<20} {reward:<15} {latency:<15} {probes:<20.1f} {reduction:<10}"
    )

# ---------------------------------------------------------------------------
# Figure 3: Probe Type Breakdown
# ---------------------------------------------------------------------------
fig3, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.0), constrained_layout=True)

methods_ordered = []
for _m in (
    "dqn",
    "simple_dqn",
    "scoring_simple_dqn",
    "scoring_enhanced_dqn",
    "conditional_dqn",
    "scoring_dqn",
):
    if _m in summary:
        methods_ordered.append(_m)
methods_ordered.extend(
    sorted(
        [m for m in summary.keys() if m not in _DQN_LIKE],
        key=lambda m: summary[m].get("total_probes", 0),
    )
)

latency_probes = [
    summary[m]["latency_probes"] / max(1, int(summary[m].get("n_selections", 1)))
    for m in methods_ordered
]
bandwidth_probes = [
    summary[m]["bandwidth_probes"] / max(1, int(summary[m].get("n_selections", 1)))
    for m in methods_ordered
]

x = np.arange(len(methods_ordered))
width = 0.35

bars_lat = ax.bar(x - width / 2, latency_probes, width, label="Latency Probes", color="#87CEEB", edgecolor="black", linewidth=0.5)
bars_bw = ax.bar(x + width / 2, bandwidth_probes, width, label="Bandwidth Probes", color="#F08080", edgecolor="black", linewidth=0.5)

ax.set_ylabel("Probes per Selection")
ax.set_title("Probe Type Breakdown")
ax.set_xticks(x)
ax.set_xticklabels([method_display_names.get(m, m) for m in methods_ordered], rotation=LABEL_ROTATION, ha=LABEL_HA)

# Add numeric labels to both Latency and Bandwidth bars
ax.bar_label(bars_lat, fmt='%.1f', padding=3, fontsize=7)
ax.bar_label(bars_bw, fmt='%.1f', padding=3, fontsize=7)

max_probe_height = max(max(latency_probes), max(bandwidth_probes))
# Extend upper boundary by 40% to comfortably house the legend and the bar text
ax.set_ylim(0, max_probe_height * 1.4) 

ax.legend(loc="upper left", ncol=1, frameon=True, facecolor="white", framealpha=0.9, edgecolor="gray")

fig3.savefig(os.path.join(run_dir, "figure3_probe_breakdown.png"), dpi=300, bbox_inches="tight")
plt.close(fig3)

# ---------------------------------------------------------------------------
# Figure 4: Multi-reward profile heatmap (conditional vs scoring DQN)
# ---------------------------------------------------------------------------
if os.path.isfile(_multi_reward_path):
    with open(_multi_reward_path, "r") as f:
        multi_reward = json.load(f)

    rows = multi_reward.get("results", [])
    if rows:
        methods_mr = multi_reward.get("methods", sorted({r["method"] for r in rows}))
        profiles_mr = multi_reward.get("profiles", sorted({r["profile"] for r in rows}))
        matrix = np.zeros((len(methods_mr), len(profiles_mr)))
        for i, method in enumerate(methods_mr):
            for j, profile in enumerate(profiles_mr):
                match = [r for r in rows if r["method"] == method and r["profile"] == profile]
                if match:
                    matrix[i, j] = match[0]["reward_mean"]

        fig4, ax = plt.subplots(
            figsize=(FULL_WIDTH, max(3.5, 0.35 * len(methods_mr) + 1.5)),
            constrained_layout=True,
        )
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(profiles_mr)))
        ax.set_yticks(np.arange(len(methods_mr)))
        ax.set_xticklabels(profiles_mr, rotation=45, ha="right")
        ax.set_yticklabels([method_display_names.get(m, m) for m in methods_mr])
        ax.set_title("Mean Reward by Method and Profile")
        for i in range(len(methods_mr)):
            for j in range(len(profiles_mr)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Reward")
        fig4.savefig(
            os.path.join(run_dir, "figure4_multi_reward_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig4)
        print("Saved figure4_multi_reward_heatmap.png")

print("Figures successfully generated with correct linear data scaling and numeric labels.")