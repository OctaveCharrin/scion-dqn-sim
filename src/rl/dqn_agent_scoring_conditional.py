"""
Reward-weight-conditioned path-scoring DQN.

Uses a weight-FiLM dueling head: reward weights modulate per-path features in the
advantage stream while the value stream depends only on topology/time context.
This avoids the failure mode of standard dueling concat, where a uniform shift in
V(s) from weight changes leaves argmax over paths unchanged.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_enhanced import (
    EnhancedPathScoringDQNAgent,
    INVALID_Q_MASK,
    PrioritizedPathScoringReplayBuffer,
    ScoringExperience,
    _pad_path_arrays,
)
from src.simulation.evaluation_env import REWARD_WEIGHT_DIM, SCORING_GLOBAL_DIM

logger = logging.getLogger(__name__)

CONDITIONAL_ARCH_WEIGHT_FILM = "weight_film"
CONDITIONAL_ARCH_LEGACY = "dueling_concat"


class WeightConditionedDuelingPathScoringDQN(nn.Module):
    """Dueling Q with FiLM modulation of path features from reward weights."""

    def __init__(
        self,
        scoring_global_dim: int,
        weight_dim: int,
        path_dim: int,
        config: EnhancedDQNConfig,
    ):
        super().__init__()
        self.scoring_global_dim = scoring_global_dim
        self.weight_dim = weight_dim
        self.path_dim = path_dim
        self.config = config
        hd = config.hidden_dim
        combined = scoring_global_dim + path_dim

        value_layers: List[nn.Module] = []
        in_dim = scoring_global_dim
        for _ in range(config.n_hidden_layers):
            value_layers.append(nn.Linear(in_dim, hd))
            if config.use_batch_norm:
                value_layers.append(nn.BatchNorm1d(hd))
            value_layers.append(nn.ReLU())
            if config.dropout_rate > 0:
                value_layers.append(nn.Dropout(config.dropout_rate))
            in_dim = hd
        value_layers.append(nn.Linear(in_dim, 1))
        self.value_stream = nn.Sequential(*value_layers)

        self.weight_film = nn.Sequential(
            nn.Linear(weight_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, 2 * path_dim),
        )

        adv_layers: List[nn.Module] = []
        in_dim = combined
        for _ in range(config.n_hidden_layers):
            adv_layers.append(nn.Linear(in_dim, hd))
            adv_layers.append(nn.ReLU())
            if config.dropout_rate > 0:
                adv_layers.append(nn.Dropout(config.dropout_rate))
            in_dim = hd
        adv_layers.append(nn.Linear(in_dim, 1))
        self.advantage_stream = nn.Sequential(*adv_layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Sequential):
            for child in module:
                WeightConditionedDuelingPathScoringDQN._init_weights(child)

    def forward(
        self,
        global_state: torch.Tensor,
        path_features: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if global_state.dim() != 2 or path_features.dim() != 3:
            raise ValueError(
                f"Expected global (B, G) and paths (B, N, P); got {global_state.shape}, {path_features.shape}"
            )
        b, g = global_state.shape
        b2, n, _p = path_features.shape
        if b != b2:
            raise ValueError(f"Batch mismatch: global B={b}, paths B={b2}")
        if g < self.scoring_global_dim + self.weight_dim:
            raise ValueError(
                f"global_state dim {g} < scoring({self.scoring_global_dim}) + weights({self.weight_dim})"
            )

        scoring = global_state[:, : self.scoring_global_dim]
        weights = global_state[:, self.scoring_global_dim : self.scoring_global_dim + self.weight_dim]

        value = self.value_stream(scoring)

        gamma_beta = self.weight_film(weights)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        mod_paths = path_features * (1.0 + gamma) + beta

        expanded = scoring.unsqueeze(1).expand(b, n, self.scoring_global_dim)
        combined = torch.cat([expanded, mod_paths], dim=-1)
        flat = combined.reshape(b * n, self.scoring_global_dim + self.path_dim)
        advantage = self.advantage_stream(flat).view(b, n)

        if path_mask is not None:
            mask_f = path_mask.float()
            adv_sum = (advantage * mask_f).sum(dim=1, keepdim=True)
            adv_count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            adv_mean = adv_sum / adv_count
        else:
            adv_mean = advantage.mean(dim=1, keepdim=True)

        q_values = value + (advantage - adv_mean)
        if path_mask is not None:
            q_values = q_values.masked_fill(~path_mask, INVALID_Q_MASK)
        return q_values


class ConditionalPathScoringDQNAgent(EnhancedPathScoringDQNAgent):
    """Path-scoring DQN with weight-FiLM conditioning (default for new training)."""

    architecture = CONDITIONAL_ARCH_WEIGHT_FILM

    def __init__(
        self,
        global_dim: int,
        path_dim: int,
        config: Optional[EnhancedDQNConfig] = None,
        *,
        scoring_global_dim: int = SCORING_GLOBAL_DIM,
        weight_dim: int = REWARD_WEIGHT_DIM,
    ):
        self.config = config or EnhancedDQNConfig()
        self.global_dim = global_dim
        self.path_dim = path_dim
        self.scoring_global_dim = scoring_global_dim
        self.weight_dim = weight_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Conditional path-scoring DQN (weight-FiLM) on %s", self.device)

        net_kw = dict(
            scoring_global_dim=scoring_global_dim,
            weight_dim=weight_dim,
            path_dim=path_dim,
            config=self.config,
        )
        self.q_network = WeightConditionedDuelingPathScoringDQN(**net_kw).to(self.device)
        self.target_network = WeightConditionedDuelingPathScoringDQN(**net_kw).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.9
        )

        if self.config.use_prioritized_replay:
            self.memory: PrioritizedPathScoringReplayBuffer | Deque[ScoringExperience] = (
                PrioritizedPathScoringReplayBuffer(
                    self.config.buffer_size,
                    self.config.alpha,
                    self.config.priority_epsilon,
                )
            )
        else:
            self.memory = deque(maxlen=self.config.buffer_size)

        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self.episodes = 0
        self.beta = self.config.beta_start
        self.losses: Deque[float] = deque(maxlen=1000)
        self.td_errors: Deque[float] = deque(maxlen=1000)


class LegacyConditionalPathScoringDQNAgent(EnhancedPathScoringDQNAgent):
    """Pre-FiLM checkpoints: full global vector in both dueling streams."""

    architecture = CONDITIONAL_ARCH_LEGACY


def infer_conditional_architecture(ckpt: Mapping[str, Any]) -> str:
    arch = str(ckpt.get("architecture", "") or "").strip()
    if arch in (CONDITIONAL_ARCH_WEIGHT_FILM, CONDITIONAL_ARCH_LEGACY):
        return arch
    sd = ckpt.get("q_network")
    if isinstance(sd, dict) and any(k.startswith("weight_film.") for k in sd):
        return CONDITIONAL_ARCH_WEIGHT_FILM
    return CONDITIONAL_ARCH_LEGACY


def load_conditional_scoring_agent(
    ckpt: Mapping[str, Any],
    *,
    config: Optional[EnhancedDQNConfig] = None,
) -> EnhancedPathScoringDQNAgent:
    """Build agent and load weights; picks FiLM vs legacy architecture from checkpoint."""
    from src.simulation.evaluation_env import (
        CONDITIONAL_SCORING_GLOBAL_DIM,
        PATH_FEATURE_DIM,
        set_conditional_weight_encoding,
    )

    cfg = config or ckpt.get("config") or EnhancedDQNConfig()
    gdim = int(ckpt.get("global_dim", CONDITIONAL_SCORING_GLOBAL_DIM))
    pdim = int(ckpt.get("path_dim", PATH_FEATURE_DIM))
    encoding = str(ckpt.get("weight_encoding", "raw"))
    set_conditional_weight_encoding(encoding)

    arch = infer_conditional_architecture(ckpt)
    if arch == CONDITIONAL_ARCH_WEIGHT_FILM:
        agent: EnhancedPathScoringDQNAgent = ConditionalPathScoringDQNAgent(
            global_dim=gdim,
            path_dim=pdim,
            config=cfg,
            scoring_global_dim=int(ckpt.get("scoring_global_dim", SCORING_GLOBAL_DIM)),
            weight_dim=int(ckpt.get("weight_dim", REWARD_WEIGHT_DIM)),
        )
    else:
        agent = LegacyConditionalPathScoringDQNAgent(
            global_dim=gdim,
            path_dim=pdim,
            config=cfg,
        )

    agent.q_network.load_state_dict(ckpt["q_network"])
    if "target_network" in ckpt:
        agent.target_network.load_state_dict(ckpt["target_network"])
    agent.epsilon = float(ckpt.get("epsilon", 0.0))
    return agent


__all__ = [
    "ConditionalPathScoringDQNAgent",
    "LegacyConditionalPathScoringDQNAgent",
    "WeightConditionedDuelingPathScoringDQN",
    "load_conditional_scoring_agent",
    "infer_conditional_architecture",
    "CONDITIONAL_ARCH_WEIGHT_FILM",
    "CONDITIONAL_ARCH_LEGACY",
]
