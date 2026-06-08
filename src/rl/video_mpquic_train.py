"""
Training + evaluation for the multipath-QUIC video path-selection agent.

This is the ``ns3`` branch analogue of ``path_selection_train.train_scoring_dqn``.
It reuses ``EnhancedPathScoringDQNAgent`` / ``EnhancedDQNConfig`` verbatim and only
swaps the environment for :class:`~src.ns3env.video_env.Ns3VideoMpquicEnv`, which
is driven by a pluggable ``DataPlane`` (mock for CI, NS-3 for real runs).

Phase 1 scope: discrete path selection, goodput+trust reward. The training loop,
replay, epsilon schedule and checkpoint format mirror the SCION scoring trainer so
later phases (rate, bitrate, conditional weights) slot in the same way.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.ns3env.dataplane import DataPlane, MockTraceConfig, MockTraceDataPlane
from src.ns3env.video_env import GLOBAL_DIM, PATH_FEATURE_DIM, Ns3VideoMpquicEnv
from src.ns3env.reward import RewardWeights
from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent

DataPlaneFactory = Callable[[], DataPlane]


def default_dataplane_factory() -> DataPlane:
    """A modest asymmetric, time-varying 3-path scenario."""
    return MockTraceDataPlane(MockTraceConfig())


@dataclass
class VideoTrainHyperparams:
    """Hyperparameters for the path-scoring agent (mirrors ScoringHyperparams)."""

    learning_rate: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.99
    buffer_size: int = 20_000
    batch_size: int = 64
    min_buffer_size: int = 500
    target_update_every: int = 100
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    tau: float = 0.05
    gradient_every: int = 4


def _make_agent(hp: VideoTrainHyperparams) -> EnhancedPathScoringDQNAgent:
    config = EnhancedDQNConfig(
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        epsilon_start=hp.epsilon_start,
        epsilon_end=hp.epsilon_end,
        epsilon_decay=hp.epsilon_decay,
        buffer_size=hp.buffer_size,
        min_buffer_size=hp.min_buffer_size,
        batch_size=hp.batch_size,
        target_update_every=hp.target_update_every,
        hidden_dim=hp.hidden_dim,
        n_hidden_layers=hp.n_hidden_layers,
        use_batch_norm=False,  # act() runs single-sample forward passes
        use_prioritized_replay=True,
        use_dueling_dqn=True,
        use_double_dqn=True,
        tau=hp.tau,
    )
    return EnhancedPathScoringDQNAgent(
        global_dim=GLOBAL_DIM, path_dim=PATH_FEATURE_DIM, config=config
    )


def train_video_dqn(
    *,
    dataplane_factory: Optional[DataPlaneFactory] = None,
    num_episodes: int = 300,
    hp: Optional[VideoTrainHyperparams] = None,
    reward_weights: Optional[RewardWeights] = None,
    seed: int = 42,
    quiet: bool = False,
    checkpoint_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], EnhancedPathScoringDQNAgent]:
    """Train the path-scoring DQN; return ``(stats, trained_agent)``."""
    hp = hp or VideoTrainHyperparams()
    dataplane_factory = dataplane_factory or default_dataplane_factory
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataplane = dataplane_factory()
    env = Ns3VideoMpquicEnv(dataplane, reward_weights=reward_weights)
    agent = _make_agent(hp)

    ep_iter: Sequence[int] = range(num_episodes)
    if not quiet:
        try:
            from tqdm import tqdm

            ep_iter = tqdm(ep_iter, desc="train_video_dqn")
        except ImportError:
            pass

    episode_rewards: List[float] = []
    losses: List[float] = []
    total_steps = 0

    for ep in ep_iter:
        state = env.reset(seed=seed + ep)
        ep_reward = 0.0
        n_steps = 0
        while True:
            action = int(agent.act(state))
            next_state, reward, done, _info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            ep_reward += reward
            n_steps += 1
            total_steps += 1
            if total_steps % hp.gradient_every == 0:
                loss = agent.replay()
                if loss is not None:
                    losses.append(float(loss))
            state = next_state
            if done:
                break
        agent.episodes += 1
        agent.epsilon = max(hp.epsilon_end, agent.epsilon * hp.epsilon_decay)
        episode_rewards.append(ep_reward / max(1, n_steps))

    stats: Dict[str, Any] = {
        "num_episodes": num_episodes,
        "episode_rewards": episode_rewards,
        "losses": losses,
        "final_epsilon": agent.epsilon,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_reward_last_decile": _tail_mean(episode_rewards),
        "hyperparams": asdict(hp),
        "global_dim": GLOBAL_DIM,
        "path_dim": PATH_FEATURE_DIM,
        "cap_mbps": env.cap_mbps,
    }

    if checkpoint_path is not None:
        payload = {
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "epsilon": agent.epsilon,
            "global_dim": GLOBAL_DIM,
            "path_dim": PATH_FEATURE_DIM,
            "hyperparams": asdict(hp),
            "cap_mbps": env.cap_mbps,
            "reward_weights": (reward_weights or RewardWeights()).to_dict(),
        }
        torch.save(payload, checkpoint_path)
        stats["checkpoint"] = str(checkpoint_path)

    if stats_path is not None:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats, agent


def _tail_mean(values: List[float], frac: float = 0.1) -> float:
    if not values:
        return 0.0
    k = max(1, int(len(values) * frac))
    return float(np.mean(values[-k:]))


def evaluate_policy(
    selector: Any,
    *,
    dataplane_factory: Optional[DataPlaneFactory] = None,
    reward_weights: Optional[RewardWeights] = None,
    episodes: int = 20,
    seed: int = 10_000,
    is_agent: bool = False,
) -> Dict[str, float]:
    """Roll out a policy over several episodes; return mean per-step reward + goodput.

    ``selector`` is either a baseline exposing ``select(obs) -> int`` (when
    ``is_agent`` is False) or an agent exposing ``act(obs, evaluate=True)``.
    """
    dataplane_factory = dataplane_factory or default_dataplane_factory
    rewards: List[float] = []
    goodputs: List[float] = []
    for ep in range(episodes):
        env = Ns3VideoMpquicEnv(dataplane_factory(), reward_weights=reward_weights)
        state = env.reset(seed=seed + ep)
        if hasattr(selector, "reset"):
            selector.reset()
        ep_r: List[float] = []
        ep_g: List[float] = []
        while True:
            if is_agent:
                action = int(selector.act(state, evaluate=True))
            else:
                action = int(selector.select(state))
            state, reward, done, info = env.step(action)
            ep_r.append(reward)
            ep_g.append(info["throughput_mbps"])
            if done:
                break
        rewards.append(float(np.mean(ep_r)))
        goodputs.append(float(np.mean(ep_g)))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_goodput_mbps": float(np.mean(goodputs)),
        "episodes": episodes,
    }
