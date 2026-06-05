"""LNCS-style figure metadata and shared plot helpers for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

COLUMN_WIDTH: float = 3.5
FULL_WIDTH: float = 7.0

METHOD_DISPLAY_NAMES: Mapping[str, str] = {
    "dqn": "DQN (Enhanced)",
    "simple_dqn": "DQN (Simple)",
    "scoring_simple_dqn": "DQN (Path-Scoring Simple)",
    "scoring_enhanced_dqn": "DQN (Path-Scoring Enhanced)",
    "conditional_dqn": "DQN (Conditional)",
    "scoring_dqn": "DQN (Path-Scoring Simple)",
    "shortest_path": "Shortest Path",
    "widest_path": "Widest Path",
    "lowest_latency": "Lowest Latency",
    "ecmp": "ECMP",
    "random": "Random",
    "scion_default": "SCION Default",
}

METHOD_COLORS: Mapping[str, str] = {
    "dqn": "#1f77b4",
    "simple_dqn": "#17becf",
    "scoring_simple_dqn": "#bcbd22",
    "scoring_enhanced_dqn": "#7f7f7f",
    "conditional_dqn": "#d62728",
    "scoring_dqn": "#bcbd22",
    "shortest_path": "#ff7f0e",
    "widest_path": "#2ca02c",
    "lowest_latency": "#d62728",
    "ecmp": "#9467bd",
    "random": "#8c564b",
    "scion_default": "#e377c2",
}


def apply_lncs_style() -> None:
    """Configure matplotlib's rcParams for LNCS-style figures."""
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"]
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 9
    rcParams["figure.titlesize"] = 12


def display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


PROFILE_DISPLAY_NAMES: Mapping[str, str] = {
    "balanced": "Balanced",
    "throughput": "Throughput",
    "trust_quality": "Trust / quality",
    "latency": "Latency",
    "low_probe_cost": "Low probe cost",
    "high_probe_cost": "High probe cost",
}

DEFAULT_METHOD_ORDER: tuple[str, ...] = (
    "conditional_dqn",
    "scoring_enhanced_dqn",
    "scoring_simple_dqn",
    "dqn",
    "simple_dqn",
    "widest_path",
    "lowest_latency",
    "shortest_path",
    "ecmp",
    "random",
    "scion_default",
)


def profile_display_name(profile: str) -> str:
    return PROFILE_DISPLAY_NAMES.get(profile, profile.replace("_", " ").title())


def summaries_for_profile(
    multi_reward: Mapping[str, Any],
    profile: str,
) -> Dict[str, Dict[str, float]]:
    """Build a ``evaluation_results.json``-style summary dict for one reward profile."""
    rows = multi_reward.get("results", [])
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row.get("profile") != profile:
            continue
        method = str(row["method"])
        out[method] = {
            "reward_mean": float(row["reward_mean"]),
            "reward_std": float(row["reward_std"]),
            "n_selections": float(row.get("n_selections", 0)),
        }
    return out


def ordered_methods(
    summary: Mapping[str, Mapping[str, float]],
    preferred: Optional[Sequence[str]] = None,
) -> List[str]:
    """Methods present in ``summary``, ordered by mean reward (desc) with optional priority."""
    pool = preferred or DEFAULT_METHOD_ORDER
    ranked = sorted(summary.keys(), key=lambda m: summary[m]["reward_mean"], reverse=True)
    head = [m for m in pool if m in summary]
    tail = [m for m in ranked if m not in head]
    return head + tail


def synthetic_reward_samples(
    mean: float,
    std: float,
    n_samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Approximate a reward distribution for box plots when only mean/std are stored."""
    gen = rng if rng is not None else np.random.default_rng(0)
    samples = gen.normal(mean, max(std, 1e-6), n_samples)
    return np.clip(samples, -1.0, 1.0)


def plot_path_reward_boxplot(
    summary: Mapping[str, Mapping[str, float]],
    output_path: Path | str,
    *,
    title: str = "Path Selection Performance",
    method_order: Optional[Sequence[str]] = None,
    n_samples: int = 336,
    rng_seed: int = 0,
    label_rotation: float = 45.0,
    label_ha: str = "right",
) -> Path:
    """
    Figure 2 style: box plot of path rewards by method (synthetic samples from mean/std).

    Returns the path written.
    """
    if not summary:
        raise ValueError("summary must contain at least one method")

    methods = ordered_methods(summary, method_order)
    rng = np.random.default_rng(rng_seed)
    reward_data: List[np.ndarray] = []
    positions = list(range(len(methods)))

    for method in methods:
        stats = summary[method]
        n = int(stats.get("n_selections") or 0)
        count = n if n > 0 else n_samples
        reward_data.append(
            synthetic_reward_samples(
                float(stats["reward_mean"]),
                float(stats.get("reward_std", 0.0)),
                count,
                rng=rng,
            )
        )

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.0), constrained_layout=True)
    bp = ax.boxplot(
        reward_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(color_for(method))
        patch.set_alpha(0.7)

    labels = [display_name(m) for m in methods]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=label_rotation, ha=label_ha)
    ax.set_ylabel("Path Reward")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-1.0, 1.2)

    for i, method in enumerate(methods):
        mean_val = float(summary[method]["reward_mean"])
        ax.text(i, 0.95, f"{mean_val:.3f}", ha="center", va="top", fontsize=6)

    out = Path(output_path)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def figure2_filename_for_profile(profile: Optional[str] = None) -> str:
    if not profile or profile == "balanced":
        return "figure2_path_reward.png"
    return f"figure2_path_reward_{profile}.png"


def generate_figure2_for_profiles(
    run_dir: Path | str,
    multi_reward: Mapping[str, Any],
    *,
    profiles: Optional[Sequence[str]] = None,
    method_order: Optional[Sequence[str]] = None,
    rng_seed: int = 0,
) -> List[Path]:
    """Write one Figure 2 PNG per reward profile under ``run_dir``."""
    run_path = Path(run_dir)
    profile_list = list(profiles) if profiles is not None else list(multi_reward.get("profiles", []))
    if not profile_list:
        profile_list = sorted({str(r["profile"]) for r in multi_reward.get("results", [])})

    order = list(method_order) if method_order else list(multi_reward.get("methods", []))
    written: List[Path] = []

    for profile in profile_list:
        summary = summaries_for_profile(multi_reward, profile)
        if not summary:
            continue
        title = f"Path Selection Performance ({profile_display_name(profile)})"
        fname = figure2_filename_for_profile(profile)
        path = plot_path_reward_boxplot(
            summary,
            run_path / fname,
            title=title,
            method_order=order or None,
            rng_seed=rng_seed + sum(ord(c) for c in profile),
        )
        written.append(path)
    return written
