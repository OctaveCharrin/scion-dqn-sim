"""DQN agents and training for path selection.

Imports are lazy (PEP 562) so that pulling in a single agent does not eagerly
import the SCION training pipeline (``path_selection_train`` -> ``src.simulation``
-> networkx, ...). This keeps the reusable RL components usable on the ``ns3``
branch with only ``torch``/``numpy`` installed, while preserving the public API:
``from src.rl import train_scoring_dqn`` still works.
"""

from typing import TYPE_CHECKING, Any

# Map exported name -> submodule providing it (resolved on first access).
_EXPORTS = {
    "EnhancedDQNAgent": "dqn_agent_enhanced",
    "EnhancedDQNConfig": "dqn_agent_enhanced",
    "ConditionalPathScoringDQNAgent": "dqn_agent_scoring_conditional",
    "EnhancedPathScoringDQNAgent": "dqn_agent_scoring_enhanced",
    "SimplePathScoringDQNAgent": "dqn_agent_scoring_simple",
    "SimpleDQNAgent": "dqn_agent_simple",
    "ScoringHyperparams": "path_selection_train",
    "train_conditional_scoring_dqn": "path_selection_train",
    "train_flat_dqn": "path_selection_train",
    "train_scoring_dqn": "path_selection_train",
    "REWARD_PROFILES": "reward_profiles",
    "RewardProfile": "reward_profiles",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # for static analysis only; not executed at runtime
    from .dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig  # noqa: F401
    from .dqn_agent_scoring_conditional import (  # noqa: F401
        ConditionalPathScoringDQNAgent,
    )
    from .dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent  # noqa: F401
    from .dqn_agent_scoring_simple import SimplePathScoringDQNAgent  # noqa: F401
    from .dqn_agent_simple import SimpleDQNAgent  # noqa: F401
    from .path_selection_train import (  # noqa: F401
        ScoringHyperparams,
        train_conditional_scoring_dqn,
        train_flat_dqn,
        train_scoring_dqn,
    )
    from .reward_profiles import REWARD_PROFILES, RewardProfile  # noqa: F401
