"""Shared pytest configuration.

Ensures the repository root is on ``sys.path`` so ``from src.<x> import ...``
works regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
