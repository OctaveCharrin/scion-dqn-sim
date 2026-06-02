"""Named reward-weight profiles and sampling for multi-objective DQN training."""

from __future__ import annotations

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


def profile_names(profiles: Optional[Sequence[RewardProfile]] = None) -> List[str]:
    pool = profiles if profiles is not None else REWARD_PROFILES
    return [p.name for p in pool]
