"""Named reward-weight profiles and sampling for multi-objective DQN training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from src.simulation.evaluation_env import RewardWeights


@dataclass(frozen=True)
class RewardProfile:
    """A named reward configuration for training or evaluation."""

    name: str
    weights: RewardWeights
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "weights": self.weights.to_dict(),
        }


# Presets spanning throughput, trust/latency sensitivity, and probe cost.
REWARD_PROFILES: List[RewardProfile] = [
    RewardProfile(
        "balanced",
        RewardWeights(w1=0.7, w2=0.3, w3=0.5, w4=0.5, w_probe=0.05),
        "Default goodput + trust mix.",
    ),
    RewardProfile(
        "throughput",
        RewardWeights(w1=0.95, w2=0.05, w3=0.35, w4=0.35, w_probe=0.05),
        "Prioritize normalized goodput (bandwidth).",
    ),
    RewardProfile(
        "trust_quality",
        RewardWeights(w1=0.15, w2=0.85, w3=0.85, w4=0.85, w_probe=0.05),
        "Prioritize low loss and low delay in the trust term.",
    ),
    RewardProfile(
        "latency",
        RewardWeights(w1=0.45, w2=0.55, w3=0.25, w4=0.95, w_probe=0.05),
        "Strong delay penalty inside trust; moderate goodput.",
    ),
    RewardProfile(
        "low_probe_cost",
        RewardWeights(w1=0.7, w2=0.3, w3=0.5, w4=0.5, w_probe=0.005),
        "Nearly ignore probing overhead.",
    ),
    RewardProfile(
        "high_probe_cost",
        RewardWeights(w1=0.7, w2=0.3, w3=0.5, w4=0.5, w_probe=0.25),
        "Heavily penalize probing.",
    ),
]

# Sharper extremes for conditional training (eval still uses REWARD_PROFILES).
DISTINCTIVE_REWARD_PROFILES: List[RewardProfile] = [
    RewardProfile(
        "bandwidth_max",
        RewardWeights(w1=1.0, w2=0.0, w3=0.05, w4=0.05, w_probe=0.05),
        "Pure goodput; ignore trust composition.",
    ),
    RewardProfile(
        "loss_averse",
        RewardWeights(w1=0.05, w2=0.95, w3=1.0, w4=0.15, w_probe=0.05),
        "Trust dominated by loss avoidance.",
    ),
    RewardProfile(
        "delay_averse",
        RewardWeights(w1=0.1, w2=0.9, w3=0.15, w4=1.0, w_probe=0.05),
        "Trust dominated by latency penalty.",
    ),
    RewardProfile(
        "balanced_extreme",
        RewardWeights(w1=0.55, w2=0.45, w3=0.55, w4=0.55, w_probe=0.05),
        "Midpoint anchor between bandwidth- and trust-heavy poles.",
    ),
    RewardProfile(
        "probe_minimal",
        RewardWeights(w1=0.7, w2=0.3, w3=0.5, w4=0.5, w_probe=0.001),
        "Almost no probe penalty (explore paths freely).",
    ),
    RewardProfile(
        "probe_averse",
        RewardWeights(w1=0.7, w2=0.3, w3=0.5, w4=0.5, w_probe=0.35),
        "Heavy probe penalty (prefer fewer / cheaper probes).",
    ),
]

_PROFILE_BY_NAME: Dict[str, RewardProfile] = {p.name: p for p in REWARD_PROFILES}


def get_profile(name: str) -> RewardProfile:
    if name not in _PROFILE_BY_NAME:
        known = ", ".join(sorted(_PROFILE_BY_NAME))
        raise KeyError(f"Unknown reward profile {name!r}; known: {known}")
    return _PROFILE_BY_NAME[name]


def sample_training_weights(
    rng: random.Random,
    profiles: Optional[Sequence[RewardProfile]] = None,
) -> RewardWeights:
    """Pick a profile uniformly (per episode) for contextual training."""
    pool = list(profiles) if profiles is not None else REWARD_PROFILES
    return rng.choice(pool).weights


def get_conditional_training_profiles() -> List[RewardProfile]:
    """Profiles used by ``train_conditional_scoring_dqn`` (env ``DQN_CONDITIONAL_PROFILES``)."""
    mode = os.environ.get("DQN_CONDITIONAL_PROFILES", "distinctive").strip().lower()
    if mode in ("legacy", "eval", "standard"):
        return list(REWARD_PROFILES)
    if mode == "all":
        merged: Dict[str, RewardProfile] = {p.name: p for p in REWARD_PROFILES}
        for p in DISTINCTIVE_REWARD_PROFILES:
            merged[p.name] = p
        return list(merged.values())
    return list(DISTINCTIVE_REWARD_PROFILES)


def stratified_training_profile_schedule(
    n_episodes: int,
    profiles: Sequence[RewardProfile],
    rng: random.Random,
) -> List[RewardProfile]:
    """Equal per-profile episode counts with shuffled order (reduces sampling noise)."""
    if not profiles:
        raise ValueError("profiles must be non-empty")
    pool = list(profiles)
    reps = (n_episodes + len(pool) - 1) // len(pool)
    schedule = (pool * reps)[:n_episodes]
    rng.shuffle(schedule)
    return schedule


def conditional_episode_multiplier() -> float:
    raw = os.environ.get("DQN_CONDITIONAL_EPISODE_MULT", "1.25").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 1.25


def profile_names(profiles: Optional[Sequence[RewardProfile]] = None) -> List[str]:
    pool = profiles if profiles is not None else REWARD_PROFILES
    return [p.name for p in pool]
