"""Global RNG seeding for reproducible training runs.

Nothing in the training loops previously seeded torch or numpy, so two runs of
the same script produced different networks and there was no way to separate
seed variance from a real architectural difference. ``set_global_seeds`` fixes
network initialization, prioritized-replay sampling and epsilon-greedy
exploration.

Deliberately *not* covered: the environment's pair / hour / weight-profile RNGs.
Those take explicit seeds at the call site and are held fixed across a seed
sweep, so every seed sees the identical stream of training contexts and only the
learning process varies.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seeds(seed: Optional[int]) -> None:
    """Seed ``random``, ``numpy`` and ``torch``. No-op when ``seed`` is None."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
