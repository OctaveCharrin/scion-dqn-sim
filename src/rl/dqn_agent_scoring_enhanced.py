"""
Path-scoring Dueling DQN with Prioritized Experience Replay and Double DQN.

States are dicts ``{"global": (G,), "paths": (N, P)}`` with variable ``N``.
Q-values are per-path scalars from a Dueling head (V from global context,
A from global + path features), not from a fixed action index.
"""

from __future__ import annotations

import logging
import random
from collections import deque, namedtuple
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.rl.dqn_agent_enhanced import EnhancedDQNConfig

logger = logging.getLogger(__name__)

INVALID_Q_MASK = -1e9

ScoringExperience = namedtuple(
    "ScoringExperience",
    ["global_state", "paths", "action", "reward", "next_global_state", "next_paths", "done"],
)


def _pad_path_arrays(
    paths_list: List[np.ndarray],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length path matrices to ``(B, max_n, P)`` with a boolean mask."""
    batch_size = len(paths_list)
    if batch_size == 0:
        raise ValueError("empty batch")
    p = int(paths_list[0].shape[1])
    max_n = max(int(x.shape[0]) for x in paths_list)
    if max_n == 0:
        raise ValueError("all path sets empty; need at least one path per transition")

    out = torch.zeros(batch_size, max_n, p, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_n, device=device, dtype=torch.bool)
    for i, arr in enumerate(paths_list):
        n = int(arr.shape[0])
        if n:
            out[i, :n] = torch.as_tensor(arr, device=device, dtype=dtype)
            mask[i, :n] = True
    return out, mask


class DuelingPathScoringDQN(nn.Module):
    """Dueling architecture that scores each candidate path from features."""

    def __init__(self, global_dim: int, path_dim: int, config: EnhancedDQNConfig):
        super().__init__()
        self.config = config
        self.global_dim = global_dim
        self.path_dim = path_dim
        hd = config.hidden_dim
        combined = global_dim + path_dim

        value_layers: List[nn.Module] = []
        in_dim = global_dim
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

    def forward(
        self,
        global_state: torch.Tensor,
        path_features: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            global_state: ``(batch, G)``
            path_features: ``(batch, max_paths, P)``
            path_mask: optional ``(batch, max_paths)`` bool; invalid slots masked in output

        Returns:
            Q-values ``(batch, max_paths)``
        """
        if global_state.dim() != 2 or path_features.dim() != 3:
            raise ValueError(
                f"Expected global (B, G) and paths (B, N, P); got {global_state.shape}, {path_features.shape}"
            )
        b, g = global_state.shape
        b2, n, _p = path_features.shape
        if b != b2:
            raise ValueError(f"Batch mismatch: global B={b}, paths B={b2}")

        value = self.value_stream(global_state)

        expanded = global_state.unsqueeze(1).expand(b, n, g)
        combined = torch.cat([expanded, path_features], dim=-1)
        flat = combined.reshape(b * n, g + self.path_dim)
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


class PrioritizedPathScoringReplayBuffer:
    """PER buffer storing dict-state transitions with batch-time path padding."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer: List[Optional[ScoringExperience]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, experience: ScoringExperience) -> None:
        max_priority = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, beta: float, device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        torch.Tensor,
    ]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        priorities = self.priorities[: self.size] + self.epsilon
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        assert all(s is not None for s in samples)

        globals_ = np.stack([s.global_state for s in samples], axis=0)
        next_globals_ = np.stack([s.next_global_state for s in samples], axis=0)

        paths_t, path_mask = _pad_path_arrays(
            [s.paths for s in samples], device, torch.float32
        )
        next_paths_t, next_path_mask = _pad_path_arrays(
            [s.next_paths for s in samples], device, torch.float32
        )

        g_t = torch.as_tensor(globals_, dtype=torch.float32, device=device)
        ng_t = torch.as_tensor(next_globals_, dtype=torch.float32, device=device)
        actions = torch.as_tensor([s.action for s in samples], dtype=torch.long, device=device)
        rewards = torch.as_tensor([s.reward for s in samples], dtype=torch.float32, device=device)
        dones = torch.as_tensor([s.done for s in samples], dtype=torch.float32, device=device)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        return (
            g_t,
            paths_t,
            path_mask,
            actions,
            rewards,
            ng_t,
            next_paths_t,
            next_path_mask,
            dones,
            indices,
            weights_t,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors) + self.epsilon

    def __len__(self) -> int:
        return self.size


class EnhancedPathScoringDQNAgent:
    """Path-scoring Dueling Double DQN with prioritized replay."""

    def __init__(
        self,
        global_dim: int,
        path_dim: int,
        config: Optional[EnhancedDQNConfig] = None,
    ):
        self.config = config or EnhancedDQNConfig()
        self.global_dim = global_dim
        self.path_dim = path_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Enhanced path-scoring DQN using device: %s", self.device)

        self.q_network = DuelingPathScoringDQN(global_dim, path_dim, self.config).to(self.device)
        self.target_network = DuelingPathScoringDQN(global_dim, path_dim, self.config).to(
            self.device
        )
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

    def remember(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: float,
    ) -> None:
        exp = ScoringExperience(
            global_state=np.asarray(state["global"], dtype=np.float32).reshape(-1),
            paths=np.asarray(state["paths"], dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_global_state=np.asarray(next_state["global"], dtype=np.float32).reshape(-1),
            next_paths=np.asarray(next_state["paths"], dtype=np.float32),
            done=float(done),
        )
        if self.config.use_prioritized_replay:
            assert isinstance(self.memory, PrioritizedPathScoringReplayBuffer)
            self.memory.push(exp)
        else:
            assert isinstance(self.memory, deque)
            self.memory.append(exp)

    def act(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> int:
        paths = np.asarray(state["paths"], dtype=np.float32)
        num_paths = int(paths.shape[0])
        if num_paths <= 0:
            raise ValueError("state['paths'] must contain at least one path row")

        if (not evaluate) and random.random() < self.epsilon:
            return random.randrange(num_paths)

        g = torch.as_tensor(state["global"], dtype=torch.float32, device=self.device).view(1, -1)
        pf = torch.as_tensor(paths, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones(1, num_paths, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q = self.q_network(g, pf, mask)
        return int(q.argmax(dim=1).item())

    def replay(self) -> Optional[float]:
        if len(self.memory) < self.config.min_buffer_size:
            return None

        if self.config.use_prioritized_replay:
            assert isinstance(self.memory, PrioritizedPathScoringReplayBuffer)
            (
                g_t,
                paths_t,
                path_mask,
                actions,
                rewards,
                ng_t,
                next_paths_t,
                next_path_mask,
                dones,
                indices,
                weights,
            ) = self.memory.sample(self.config.batch_size, self.beta, self.device)
        else:
            assert isinstance(self.memory, deque)
            batch = random.sample(self.memory, self.config.batch_size)
            g_t = torch.as_tensor(
                np.stack([e.global_state for e in batch]), dtype=torch.float32, device=self.device
            )
            ng_t = torch.as_tensor(
                np.stack([e.next_global_state for e in batch]),
                dtype=torch.float32,
                device=self.device,
            )
            paths_t, path_mask = _pad_path_arrays([e.paths for e in batch], self.device)
            next_paths_t, next_path_mask = _pad_path_arrays(
                [e.next_paths for e in batch], self.device
            )
            actions = torch.as_tensor([e.action for e in batch], dtype=torch.long, device=self.device)
            rewards = torch.as_tensor(
                [e.reward for e in batch], dtype=torch.float32, device=self.device
            )
            dones = torch.as_tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
            indices = None
            weights = torch.ones(self.config.batch_size, device=self.device)

        current_q = self.q_network(g_t, paths_t, path_mask).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.use_double_dqn:
                next_q_online = self.q_network(ng_t, next_paths_t, next_path_mask)
                neg = torch.finfo(next_q_online.dtype).min / 4
                next_actions = next_q_online.masked_fill(~next_path_mask, neg).argmax(dim=1)
                next_q = (
                    self.target_network(ng_t, next_paths_t, next_path_mask)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                next_q_all = self.target_network(ng_t, next_paths_t, next_path_mask)
                neg = torch.finfo(next_q_all.dtype).min / 4
                next_q = next_q_all.masked_fill(~next_path_mask, neg).max(dim=1)[0]

            target_q = rewards + self.config.gamma * next_q * (1.0 - dones)

        td_errors = torch.abs(current_q - target_q)
        element_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        if self.steps % self.config.target_update_every == 0:
            self.scheduler.step()

        if self.config.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        if self.steps % self.config.target_update_every == 0:
            self._soft_update_target_network()

        self.steps += 1
        self.losses.append(float(loss.item()))
        self.td_errors.extend(td_errors.detach().cpu().numpy().tolist())

        if self.config.use_prioritized_replay:
            self.beta = min(
                self.config.beta_end,
                self.beta
                + (self.config.beta_end - self.config.beta_start)
                / self.config.beta_annealing_steps,
            )

        return float(loss.item())

    def _soft_update_target_network(self) -> None:
        tau = self.config.tau
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def update_epsilon(self) -> None:
        if self.episodes < self.config.epsilon_decay_steps:
            decay = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
            self.epsilon = self.config.epsilon_start - decay * self.episodes
        else:
            self.epsilon = self.config.epsilon_end
        self.episodes += 1

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "episodes": self.episodes,
                "beta": self.beta,
                "config": self.config,
                "global_dim": self.global_dim,
                "path_dim": self.path_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.beta = checkpoint.get("beta", self.config.beta_start)

    def get_statistics(self) -> Dict[str, float]:
        return {
            "steps": float(self.steps),
            "episodes": float(self.episodes),
            "epsilon": float(self.epsilon),
            "buffer_size": float(len(self.memory)),
            "avg_loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "avg_td_error": float(np.mean(self.td_errors)) if self.td_errors else 0.0,
            "beta": float(self.beta) if self.config.use_prioritized_replay else 1.0,
        }
