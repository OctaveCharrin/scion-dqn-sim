#!/usr/bin/env python3
"""Training curves for the single-path agents (thesis appendix, app:extra)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.pipeline.intent_cond_figures import (
    plot_per_profile_training_curves,
    plot_training_curves,
)
from src.pipeline.run_dirs import resolve_run_dir

# stats filename -> label used in the figure legend
STATS_FILES: Dict[str, str] = {
    "training_stats.json": "Flat DQN",
    "dqn_scoring_enhanced_training_stats.json": "Scoring DQN (uncond.)",
    "dqn_scoring_simple_training_stats.json": "Scoring DQN (simple)",
    "dqn_conditional_value_concat_training_stats.json": "Value-Concat",
    "dqn_conditional_concat_training_stats.json": "Two-Stream-Concat",
    "dqn_conditional_training_stats.json": "Conditional-FiLM",
}
CONDITIONAL_STATS = {
    "dqn_conditional_training_stats.json": "film",
    "dqn_conditional_concat_training_stats.json": "concat2stream",
    "dqn_conditional_value_concat_training_stats.json": "valueconcat",
}


def summarize(path: Path, window: int = 50) -> Dict[str, float]:
    with open(path) as f:
        stats = json.load(f)
    rewards = np.asarray(stats.get("episode_rewards", []), dtype=float)
    losses = np.asarray(stats.get("losses", []), dtype=float)
    out = {
        "episodes": int(rewards.size),
        "reward_first": (
            float(rewards[:window].mean()) if rewards.size else float("nan")
        ),
        "reward_last": (
            float(rewards[-window:].mean()) if rewards.size else float("nan")
        ),
        "reward_mean": float(rewards.mean()) if rewards.size else float("nan"),
        "final_epsilon": float(stats.get("final_epsilon", float("nan"))),
        "n_losses": int(losses.size),
        "loss_first": (
            float(losses[: window * 4].mean()) if losses.size else float("nan")
        ),
        "loss_last": (
            float(losses[max(0, losses.size - window * 4):].mean())
            if losses.size
            else float("nan")
        ),
    }
    out["reward_gain"] = out["reward_last"] - out["reward_first"]
    out["loss_trend"] = out["loss_last"] - out["loss_first"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=25)
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=(),
        metavar="LABEL",
        help="Legend labels to omit, e.g. 'Conditional-FiLM' for a variant the "
        "thesis no longer reports.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "gap" / "training")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    excluded = set(args.exclude)
    present = {
        label: run_path / fname
        for fname, label in STATS_FILES.items()
        if (run_path / fname).is_file() and label not in excluded
    }
    if not present:
        raise SystemExit(f"No *_training_stats.json found under {run_path}")

    summary = {}
    print(f"Training curves on {run_path}")
    print(
        f"{'agent':<24} {'eps':>5} {'first50':>9} {'last50':>9} {'gain':>8} "
        f"{'loss trend':>11} {'final eps':>10}"
    )
    for label, path in present.items():
        s = summarize(path, args.smooth * 2)
        summary[label] = {"stats_file": str(path), **s}
        print(
            f"{label:<24} {s['episodes']:>5} {s['reward_first']:>9.3f} "
            f"{s['reward_last']:>9.3f} {s['reward_gain']:>+8.3f} "
            f"{s['loss_trend']:>+11.4f} {s['final_epsilon']:>10.4f}"
        )

    out = {
        "fig_training_curves": str(
            plot_training_curves(
                present, fig_dir / "fig_training_curves.png", smooth=args.smooth
            )
        )
    }
    for fname, tag in CONDITIONAL_STATS.items():
        p = run_path / fname
        if p.is_file() and STATS_FILES[fname] not in excluded:
            try:
                out[f"fig_per_intent_{tag}"] = str(
                    plot_per_profile_training_curves(
                        p, fig_dir / f"fig_training_per_intent_{tag}.png"
                    )
                )
            except ValueError as exc:
                print(f"  skipped per-intent curves for {fname}: {exc}")

    json_path = out_dir / "training_summary.json"
    with open(json_path, "w") as f:
        json.dump({"run_dir": str(run_path), "agents": summary}, f, indent=2)
    for key, path in out.items():
        print(f"  saved: {path}")
    print(f"  saved: {json_path}")


if __name__ == "__main__":
    main()
