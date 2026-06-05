#!/usr/bin/env python3
"""Diagnose conditional vs scoring DQN: action agreement and weight sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.pipeline.run_dirs import resolve_run_dir
from src.rl.dqn_agent_scoring_conditional import load_conditional_scoring_agent
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.reward_profiles import REWARD_PROFILES
from src.simulation.evaluation_env import PATH_FEATURE_DIM, SCORING_GLOBAL_DIM
from src.simulation.run_context import load_run_context, make_env

EVAL_HOURS = list(range(14 * 24, 14 * 24 + 24))


def _load_scoring_enhanced(run_path: Path) -> EnhancedPathScoringDQNAgent | None:
    import torch

    path = run_path / "dqn_scoring_enhanced_model.pth"
    if not path.is_file():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config") or EnhancedDQNConfig()
    agent = EnhancedPathScoringDQNAgent(
        global_dim=int(ckpt.get("global_dim", SCORING_GLOBAL_DIM)),
        path_dim=int(ckpt.get("path_dim", PATH_FEATURE_DIM)),
        config=cfg,
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.epsilon = 0.0
    return agent


def _load_conditional(run_path: Path):
    import torch

    path = run_path / "dqn_conditional_scoring_model.pth"
    if not path.is_file():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return load_conditional_scoring_agent(ckpt)


def analyze(
    run_path: Path,
    *,
    max_pairs: int = 16,
    max_hours: int = 24,
) -> Dict[str, Any]:
    topology, path_store, link_states, pair_pool, _ = load_run_context(run_path)
    pairs = pair_pool[: min(len(pair_pool), max_pairs)]
    hours = EVAL_HOURS[:max_hours]

    cond = _load_conditional(run_path)
    scoring = _load_scoring_enhanced(run_path)
    if cond is None:
        raise FileNotFoundError("Missing dqn_conditional_scoring_model.pth")

    env = make_env(
        topology, path_store, link_states, pairs, episode_length=1, rng_seed=0
    )

    profile_actions: Dict[str, List[int]] = {p.name: [] for p in REWARD_PROFILES}
    agree_with_scoring = 0
    total = 0
    cross_profile_disagree = 0
    cross_profile_total = 0

    for hour in hours:
        for src, dst in pairs:
            env.reset(source_as=src, dest_as=dst, hour_idx=hour)
            if not env.available_paths:
                continue
            n_paths = len(env.available_paths)
            scoring_state = env.observe_scoring()
            scoring_action = (
                int(scoring.act(scoring_state, evaluate=True)) if scoring else None
            )

            per_state_actions: List[int] = []
            for profile in REWARD_PROFILES:
                env.reward_weights = profile.weights
                state = env.observe_scoring_conditional()
                if state["paths"].shape[0] == 0:
                    continue
                a = int(cond.act(state, evaluate=True))
                if a >= n_paths:
                    a = 0
                profile_actions[profile.name].append(a)
                per_state_actions.append(a)

                if scoring_action is not None:
                    total += 1
                    if a == scoring_action:
                        agree_with_scoring += 1

            if len(per_state_actions) == len(REWARD_PROFILES):
                cross_profile_total += 1
                if len(set(per_state_actions)) > 1:
                    cross_profile_disagree += 1

    return {
        "run": str(run_path),
        "n_pairs": len(pairs),
        "n_hours": len(hours),
        "conditional_vs_scoring_agreement": (
            agree_with_scoring / total if total else None
        ),
        "cross_profile_action_diversity_rate": (
            cross_profile_disagree / cross_profile_total if cross_profile_total else None
        ),
        "profile_action_histogram": {
            name: _hist(profile_actions[name]) for name in profile_actions
        },
        "has_scoring_enhanced": scoring is not None,
    }


def _hist(actions: List[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for a in actions:
        out[str(a)] = out.get(str(a), 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run directory name or path (default: latest)",
    )
    parser.add_argument("--run", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    run_id = args.run_dir or args.run
    if run_id:
        run_path = Path(run_id)
        if not run_path.is_absolute():
            run_path = Path.cwd() / run_path
    else:
        run_path = Path(resolve_run_dir())
    report = analyze(run_path, max_pairs=args.max_pairs)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.write_text(text)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
