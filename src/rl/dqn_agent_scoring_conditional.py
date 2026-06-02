"""
Reward-weight-conditioned path-scoring DQN.

Uses the same Dueling Double DQN architecture as
:class:`~src.rl.dqn_agent_scoring_enhanced.EnhancedPathScoringDQNAgent`, but the
global state includes normalized reward weights so one policy can target
different composite objectives at inference time.
"""

from __future__ import annotations

from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent

ConditionalPathScoringDQNAgent = EnhancedPathScoringDQNAgent

__all__ = ["ConditionalPathScoringDQNAgent"]
