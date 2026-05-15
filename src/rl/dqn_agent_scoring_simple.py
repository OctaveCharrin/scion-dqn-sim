"""Path-scoring (parameterized action space) DQN for variable candidate path sets.

Each path is scored from learned features; the argmax over per-path Q-values is
the discrete action. States are dicts ``{"global": (G,), "paths": (N, P)}`` with
``N`` changing per step.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PathScoringDQN(nn.Module):
    """Scores each path from global context + path features."""

    def __init__(self, global_dim: int, path_dim: int, hidden_dim: int = 128):
        super().__init__()
        combined = global_dim + path_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor, path_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: (batch, G)
            path_features: (batch, num_paths, P)

        Returns:
            Tensor of shape (batch, num_paths) with one scalar Q per path.
        """
        if global_state.dim() != 2 or path_features.dim() != 3:
            raise ValueError(
                "Expected global_state (B, G) and path_features (B, N, P); "
                f"got {global_state.shape} and {path_features.shape}"
            )
        b, g = global_state.shape
        b2, n, p = path_features.shape
        if b != b2:
            raise ValueError(f"Batch mismatch: global B={b}, paths B={b2}")
        expanded = global_state.unsqueeze(1).expand(b, n, g)
        combined = torch.cat([expanded, path_features], dim=-1)
        return self.mlp(combined).squeeze(-1)


def _pad_path_batch(
    paths_list: List[np.ndarray], device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length path matrices to (B, max_n, P) with a boolean validity mask."""
    batch_size = len(paths_list)
    if batch_size == 0:
        raise ValueError("empty batch")
    p = int(paths_list[0].shape[1])
    max_n = max(int(x.shape[0]) for x in paths_list)
    if max_n == 0:
        raise ValueError("all path sets empty; need at least one path per sample for training")

    out = torch.zeros(batch_size, max_n, p, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_n, device=device, dtype=torch.bool)
    for i, arr in enumerate(paths_list):
        n = int(arr.shape[0])
        if n:
            out[i, :n] = torch.as_tensor(arr, device=device, dtype=dtype)
            mask[i, :n] = True
    return out, mask


class SimplePathScoringDQNAgent:
    """DQN agent with dict states, per-path Q heads, and padded replay batches."""

    def __init__(
        self,
        global_dim: int,
        path_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_every: int = 100,
        min_buffer_size: int = 500,
        hidden_dim: int = 128,
    ):
        self.global_dim = global_dim
        self.path_dim = path_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.min_buffer_size = min_buffer_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = PathScoringDQN(global_dim, path_dim, hidden_dim).to(self.device)
        self.target_network = PathScoringDQN(global_dim, path_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.memory: Deque[
            Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, float]
        ] = deque(maxlen=buffer_size)
        self.steps = 0
        self.episodes = 0

    def remember(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: float,
    ) -> None:
        g = np.asarray(state["global"], dtype=np.float32).reshape(-1)
        p = np.asarray(state["paths"], dtype=np.float32)
        ng = np.asarray(next_state["global"], dtype=np.float32).reshape(-1)
        np_ = np.asarray(next_state["paths"], dtype=np.float32)
        self.memory.append((g, p, int(action), float(reward), ng, np_, float(done)))

    def act(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> int:
        paths = np.asarray(state["paths"], dtype=np.float32)
        num_paths = int(paths.shape[0])
        if num_paths <= 0:
            raise ValueError("state['paths'] must contain at least one path row to select an action")

        if (not evaluate) and random.random() < self.epsilon:
            return random.randrange(num_paths)

        g = torch.as_tensor(state["global"], dtype=torch.float32, device=self.device).view(1, -1)
        pf = torch.as_tensor(paths, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_network(g, pf)
        return int(torch.argmax(q, dim=1).item())

    def replay(self) -> float | None:
        if len(self.memory) < self.min_buffer_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        globals_, paths_, actions, rewards, next_globals_, next_paths_, dones = zip(*batch)

        g_t = torch.as_tensor(np.stack(globals_, axis=0), dtype=torch.float32, device=self.device)
        ng_t = torch.as_tensor(np.stack(next_globals_, axis=0), dtype=torch.float32, device=self.device)

        paths_t, paths_mask = _pad_path_batch(list(paths_), self.device, torch.float32)
        next_paths_t, next_paths_mask = _pad_path_batch(list(next_paths_), self.device, torch.float32)

        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_all = self.q_network(g_t, paths_t)
        current_q = current_q_all.gather(1, actions_t)

        with torch.no_grad():
            next_q_all = self.target_network(ng_t, next_paths_t)
            neg_large = torch.finfo(next_q_all.dtype).min / 4
            next_q_masked = next_q_all.masked_fill(~next_paths_mask, neg_large)
            next_max_q = next_q_masked.max(dim=1, keepdim=True)[0]

        target_q = rewards_t + self.gamma * next_max_q * (1.0 - dones_t)
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

        return float(loss.item())
