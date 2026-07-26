"""Shared CLI helpers for evaluation step-04 DQN training scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.pipeline.run_dirs import resolve_run_dir
from src.rl.dqn_agent_scoring_conditional import (
    CONDITIONAL_ARCH_LEGACY,
    CONDITIONAL_ARCH_VALUE_CONCAT,
    CONDITIONAL_ARCH_WEIGHT_FILM,
)
from src.rl.path_selection_train import (
    FlatVariant,
    ScoringHyperparams,
    ScoringVariant,
    train_conditional_scoring_dqn,
    train_flat_dqn,
    train_scoring_dqn,
)
from src.simulation.run_context import load_run_context

_SCORING_CHECKPOINTS: Dict[ScoringVariant, str] = {
    "simple": "dqn_scoring_simple_model.pth",
    "enhanced": "dqn_scoring_enhanced_model.pth",
}

_SCORING_STATS: Dict[ScoringVariant, str] = {
    "simple": "dqn_scoring_simple_training_stats.json",
    "enhanced": "dqn_scoring_enhanced_training_stats.json",
}


def episodes_from_env() -> Optional[int]:
    raw = os.environ.get("DQN_TRAIN_EPISODES", "").strip()
    return int(raw) if raw.isdigit() else None


def build_train_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Evaluation run directory (default: latest run_* in cwd).",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file overriding ScoringHyperparams fields.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint output path.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Override training stats JSON path (scoring trainers only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch/numpy/random for a reproducible run (also DQN_SEED). "
        "Environment pair/hour/profile RNGs stay fixed so only the learning "
        "process varies across seeds.",
    )
    return parser


def resolve_train_run_path(run_dir: Optional[str]) -> Path:
    return Path(run_dir or resolve_run_dir())


def scoring_hyperparams(config_json: Optional[Path] = None) -> ScoringHyperparams:
    hp = ScoringHyperparams.from_env()
    if config_json and config_json.is_file():
        with open(config_json) as f:
            hp = ScoringHyperparams.from_dict(json.load(f))
    return hp


def print_training_summary(
    stats: Dict,
    *,
    title: str,
    run_path: Path,
    extra_lines: Optional[List[str]] = None,
) -> None:
    print(f"\n{title} on {run_path}")
    print(f"  avg train reward: {stats['avg_reward']:.3f}")
    for line in extra_lines or []:
        print(f"  {line}")
    ckpt = stats.get("checkpoint")
    if ckpt:
        print(f"  saved: {ckpt}")


def run_flat_training(
    variant: FlatVariant, title: str, argv: Optional[List[str]] = None
) -> None:
    args = build_train_parser(title).parse_args(argv)
    run_path = resolve_train_run_path(args.run_dir)
    stats = train_flat_dqn(
        run_path,
        variant,
        num_episodes=episodes_from_env(),
        checkpoint_path=args.checkpoint,
        stats_path=args.stats_json,
        seed=args.seed,
    )
    print_training_summary(stats, title=title, run_path=run_path)


def run_scoring_training(
    variant: ScoringVariant,
    title: str,
    argv: Optional[List[str]] = None,
) -> None:
    args = build_train_parser(title).parse_args(argv)
    run_path = resolve_train_run_path(args.run_dir)
    hp = scoring_hyperparams(args.config_json)
    if args.seed is not None:
        hp.seed = args.seed
    ckpt = args.checkpoint or run_path / _SCORING_CHECKPOINTS[variant]
    stats_path = args.stats_json or run_path / _SCORING_STATS[variant]
    stats = train_scoring_dqn(
        run_path,
        variant,
        hp,
        num_episodes=episodes_from_env(),
        checkpoint_path=ckpt,
        stats_path=stats_path,
    )
    print_training_summary(stats, title=title, run_path=run_path)


def _resolve_conditional_architecture(architecture: Optional[str]) -> str:
    """Resolve the conditioning architecture from arg or ``DQN_CONDITIONAL_ARCH`` env.

    Accepts ``film`` (FiLM modulation, default), ``value_concat`` (weights into the
    value stream only -- the naive ablation), and ``concat``/``legacy`` (full
    global vector in both dueling streams).
    """
    raw = (
        (architecture or os.environ.get("DQN_CONDITIONAL_ARCH", "film")).strip().lower()
    )
    if raw in ("film", "weight_film", CONDITIONAL_ARCH_WEIGHT_FILM):
        return CONDITIONAL_ARCH_WEIGHT_FILM
    if raw in ("value_concat", "value_only_concat", CONDITIONAL_ARCH_VALUE_CONCAT):
        return CONDITIONAL_ARCH_VALUE_CONCAT
    if raw in ("concat", "dueling_concat", "legacy", CONDITIONAL_ARCH_LEGACY):
        return CONDITIONAL_ARCH_LEGACY
    raise ValueError(
        f"Unknown DQN_CONDITIONAL_ARCH {raw!r}; expected 'film', "
        f"'value_concat', or 'concat'."
    )


def run_conditional_training(
    argv: Optional[List[str]] = None,
    *,
    architecture: Optional[str] = None,
) -> None:
    arch = _resolve_conditional_architecture(architecture)
    _kinds = {
        CONDITIONAL_ARCH_WEIGHT_FILM: "FiLM",
        CONDITIONAL_ARCH_VALUE_CONCAT: "value-only concat",
        CONDITIONAL_ARCH_LEGACY: "concat",
    }
    kind = _kinds[arch]
    title = f"Conditional path-scoring DQN ({kind})"
    args = build_train_parser(title).parse_args(argv)
    run_path = resolve_train_run_path(args.run_dir)
    hp = scoring_hyperparams(args.config_json)
    if args.seed is not None:
        hp.seed = args.seed
    stats = train_conditional_scoring_dqn(
        run_path,
        hp,
        num_episodes=episodes_from_env(),
        checkpoint_path=args.checkpoint,
        stats_path=args.stats_json,
        architecture=arch,
    )
    profiles = stats.get("training_profiles") or []
    extra = [f"profiles: {', '.join(profiles)}"] if profiles else None
    print_training_summary(stats, title=title, run_path=run_path, extra_lines=extra)


def run_all_dqn_models(run_path: Path) -> None:
    """Train every DQN variant in one process (single ``link_states.pkl`` load)."""
    print(f"\nTraining all DQN models on {run_path} (shared run context)")
    ctx = load_run_context(run_path)
    n_episodes = episodes_from_env()
    hp = ScoringHyperparams.from_env()

    jobs: List[tuple[str, Callable[[], Dict]]] = [
        (
            "Enhanced flat DQN",
            lambda: train_flat_dqn(
                run_path, "enhanced", num_episodes=n_episodes, run_context=ctx
            ),
        ),
        (
            "Simple flat DQN",
            lambda: train_flat_dqn(
                run_path, "simple", num_episodes=n_episodes, run_context=ctx
            ),
        ),
        (
            "Simple path-scoring DQN",
            lambda: train_scoring_dqn(
                run_path, "simple", hp, num_episodes=n_episodes, run_context=ctx
            ),
        ),
        (
            "Enhanced path-scoring DQN",
            lambda: train_scoring_dqn(
                run_path, "enhanced", hp, num_episodes=n_episodes, run_context=ctx
            ),
        ),
        (
            "Conditional path-scoring DQN",
            lambda: train_conditional_scoring_dqn(
                run_path, hp, num_episodes=n_episodes, run_context=ctx
            ),
        ),
    ]

    for title, train_fn in jobs:
        stats = train_fn()
        extra: Optional[List[str]] = None
        if title.startswith("Conditional"):
            profiles = stats.get("training_profiles") or []
            if profiles:
                extra = [f"profiles: {', '.join(profiles)}"]
        print_training_summary(stats, title=title, run_path=run_path, extra_lines=extra)

    print("\nAll models trained.")
