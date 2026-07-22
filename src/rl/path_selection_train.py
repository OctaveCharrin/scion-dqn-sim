"""Training loops for flat and path-scoring DQNs on evaluation run directories."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import (
    CONDITIONAL_ARCH_LEGACY,
    CONDITIONAL_ARCH_VALUE_CONCAT,
    CONDITIONAL_ARCH_WEIGHT_FILM,
    ConditionalPathScoringDQNAgent,
    LegacyConditionalPathScoringDQNAgent,
    ValueConcatConditionalPathScoringDQNAgent,
)
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
from src.rl.dqn_agent_scoring_simple import SimplePathScoringDQNAgent
from src.rl.dqn_agent_simple import SimpleDQNAgent
from src.rl.reward_profiles import (
    conditional_episode_multiplier,
    get_conditional_training_profiles,
    stratified_training_profile_schedule,
)
from src.simulation.evaluation_env import (
    CONDITIONAL_SCORING_GLOBAL_DIM,
    FLAT_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    REWARD_WEIGHT_DIM,
    SCORING_GLOBAL_DIM,
    RewardWeights,
    get_conditional_weight_encoding,
    set_conditional_weight_encoding,
)
from src.simulation.run_context import compute_action_dim, load_run_context, make_env

FlatVariant = Literal["enhanced", "simple"]
ScoringVariant = Literal["simple", "enhanced"]

TRAINING_HOURS = list(range(14 * 24))
EPISODE_LENGTH = 24


def _training_pair_cap(pair_pool_size: int) -> int:
    """Cap pair-pool size when scaling training steps (full mesh ≠ more RL steps)."""
    raw = os.environ.get("DQN_TRAIN_PAIR_CAP", "64").strip()
    if raw.isdigit():
        cap = int(raw)
        return min(pair_pool_size, max(1, cap))
    return min(pair_pool_size, 64)


def _gradient_every() -> int:
    raw = os.environ.get("DQN_GRADIENT_EVERY", "4").strip()
    return max(1, int(raw)) if raw.isdigit() else 4


def _target_episodes(pair_pool_size: int, episode_length: int = EPISODE_LENGTH) -> int:
    effective_pairs = _training_pair_cap(pair_pool_size)
    target_steps = max(2_000, min(20_000, 200 * effective_pairs))
    n = max(50, target_steps // episode_length)
    env_eps = os.environ.get("DQN_TRAIN_EPISODES", "").strip()
    if env_eps.isdigit():
        return int(env_eps)
    return n


@dataclass
class ScoringHyperparams:
    learning_rate: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10_000
    batch_size: int = 32
    min_buffer_size: int = 500
    target_update_every: int = 100
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    tau: float = 0.05
    use_prioritized_replay: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringHyperparams":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_env(cls) -> "ScoringHyperparams":
        hp = cls()
        mapping = {
            "DQN_LR": ("learning_rate", float),
            "DQN_GAMMA": ("gamma", float),
            "DQN_EPSILON_END": ("epsilon_end", float),
            "DQN_EPSILON_DECAY": ("epsilon_decay", float),
            "DQN_BATCH_SIZE": ("batch_size", int),
            "DQN_MIN_BUFFER": ("min_buffer_size", int),
            "DQN_TARGET_UPDATE": ("target_update_every", int),
            "DQN_HIDDEN_DIM": ("hidden_dim", int),
            "DQN_N_HIDDEN": ("n_hidden_layers", int),
            "DQN_TAU": ("tau", float),
        }
        for env_key, (attr, cast) in mapping.items():
            val = os.environ.get(env_key, "").strip()
            if val:
                setattr(hp, attr, cast(val))
        return hp


def train_flat_dqn(
    run_path: Path,
    variant: FlatVariant,
    *,
    num_episodes: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    run_context: Optional[Tuple] = None,
) -> Dict[str, Any]:
    if run_context is None:
        run_context = load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, goodput_cap = run_context
    with open(run_path / "selected_pair.json", "r") as f:
        selected_pair = json.load(f)
    action_dim = compute_action_dim(path_store, selected_pair)
    reward_weights = RewardWeights()

    env = make_env(
        topology_data,
        path_store,
        link_states,
        pair_pool,
        episode_length=EPISODE_LENGTH,
        rng_seed=42,
        reward_weights=reward_weights,
    )

    if variant == "enhanced":
        config = EnhancedDQNConfig(
            learning_rate=1e-3,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10_000,
            min_buffer_size=500,
            batch_size=32,
            target_update_every=100,
            hidden_dim=128,
            n_hidden_layers=2,
            use_batch_norm=False,
            use_prioritized_replay=False,
            use_dueling_dqn=True,
            use_double_dqn=True,
            use_action_masking=True,
            tau=0.05,
        )
        agent = EnhancedDQNAgent(FLAT_GLOBAL_DIM, action_dim, config)
        ckpt_name = "dqn_model.pth"
        stats_name = "training_stats.json"
    else:
        config = None
        agent = SimpleDQNAgent(
            state_dim=FLAT_GLOBAL_DIM,
            action_dim=action_dim,
            learning_rate=1e-3,
            gamma=0.95,
        )
        ckpt_name = "dqn_simple_model.pth"
        stats_name = "dqn_simple_training_stats.json"

    n_episodes = num_episodes or _target_episodes(len(pair_pool))
    grad_every = _gradient_every()
    pair_rng = random.Random(123)
    hour_rng = random.Random(456)
    episode_rewards: List[float] = []
    episode_probes: List[int] = []
    losses: List[float] = []
    total_steps = 0

    for _ep in tqdm(range(n_episodes), desc=f"train_flat_{variant}"):
        pair = pair_rng.choice(pair_pool)
        start_hour = hour_rng.choice(TRAINING_HOURS)
        env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=start_hour)
        state = env.observe_flat()
        mask = env.action_mask(action_dim)
        ep_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            if variant == "enhanced":
                action = int(agent.act(state, action_mask=mask))
            else:
                action = int(agent.act(state))
            if not mask[action]:
                valid = np.where(mask)[0]
                action = int(valid[0]) if len(valid) > 0 else 0

            reward, done, _info = env.apply_action(action, probe="full")
            next_state = env.observe_flat()
            next_mask = env.action_mask(action_dim)
            ep_reward += reward

            if variant == "enhanced":
                agent.remember(state, action, reward, next_state, done, mask, next_mask)
            else:
                agent.remember(state, action, reward, next_state, done)

            total_steps += 1
            if (
                len(agent.memory)
                >= (config.min_buffer_size if config else agent.min_buffer_size)
                and total_steps % grad_every == 0
            ):
                loss = agent.replay()
                if loss is not None:
                    losses.append(float(loss))

            state = next_state
            mask = next_mask
            if done:
                break

        if variant == "enhanced":
            agent.epsilon = max(
                config.epsilon_end, agent.epsilon * config.epsilon_decay
            )
        else:
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        agent.episodes += 1
        episode_rewards.append(ep_reward / max(1, EPISODE_LENGTH))
        episode_probes.append(int(env.num_latency_probes + env.num_bandwidth_probes))

    ckpt = checkpoint_path or (run_path / ckpt_name)
    payload: Dict[str, Any] = {
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
        "steps": agent.steps,
        "episodes": agent.episodes,
        "state_dim": FLAT_GLOBAL_DIM,
        "action_dim": action_dim,
        "goodput_cap_mbps": goodput_cap,
        "reward_weights": reward_weights.to_dict(),
        "pair_pool": [list(p) for p in pair_pool],
        "variant": variant,
    }
    if variant == "enhanced":
        payload["optimizer"] = agent.optimizer.state_dict()
        payload["scheduler"] = agent.scheduler.state_dict()
        payload["config"] = config
    else:
        payload["optimizer"] = agent.optimizer.state_dict()

    torch.save(payload, ckpt)

    stats = {
        "variant": variant,
        "num_episodes": n_episodes,
        "episode_length_hours": EPISODE_LENGTH,
        "total_steps": total_steps,
        "episode_rewards": episode_rewards,
        "episode_probes": episode_probes,
        "losses": losses,
        "final_epsilon": agent.epsilon,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_probes_per_episode": (
            float(np.mean(episode_probes)) if episode_probes else 0.0
        ),
        "reward_weights": reward_weights.to_dict(),
        "goodput_cap_mbps": goodput_cap,
        "pair_pool_size": len(pair_pool),
        "action_dim": action_dim,
        "checkpoint": str(ckpt),
    }
    out_stats = stats_path or (run_path / stats_name)
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def train_scoring_dqn(
    run_path: Path,
    variant: ScoringVariant,
    hp: ScoringHyperparams,
    *,
    num_episodes: Optional[int] = None,
    episode_length: int = EPISODE_LENGTH,
    checkpoint_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    env_seed: int = 42,
    pair_rng_seed: int = 123,
    hour_rng_seed: int = 456,
    quiet: bool = False,
    run_context: Optional[Tuple] = None,
) -> Dict[str, Any]:
    if run_context is None:
        run_context = load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, goodput_cap = run_context
    reward_weights = RewardWeights()

    env = make_env(
        topology_data,
        path_store,
        link_states,
        pair_pool,
        episode_length=episode_length,
        rng_seed=env_seed,
        reward_weights=reward_weights,
    )

    if variant == "simple":
        agent = SimplePathScoringDQNAgent(
            global_dim=SCORING_GLOBAL_DIM,
            path_dim=PATH_FEATURE_DIM,
            learning_rate=hp.learning_rate,
            gamma=hp.gamma,
            epsilon_start=hp.epsilon_start,
            epsilon_end=hp.epsilon_end,
            epsilon_decay=hp.epsilon_decay,
            buffer_size=hp.buffer_size,
            batch_size=hp.batch_size,
            target_update_every=hp.target_update_every,
            min_buffer_size=hp.min_buffer_size,
            hidden_dim=hp.hidden_dim,
        )
        config = None
    else:
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
            use_batch_norm=False,
            use_prioritized_replay=hp.use_prioritized_replay,
            use_dueling_dqn=True,
            use_double_dqn=True,
            tau=hp.tau,
        )
        agent = EnhancedPathScoringDQNAgent(
            global_dim=SCORING_GLOBAL_DIM,
            path_dim=PATH_FEATURE_DIM,
            config=config,
        )

    n_episodes = num_episodes or _target_episodes(len(pair_pool), episode_length)
    grad_every = _gradient_every()
    pair_rng = random.Random(pair_rng_seed)
    hour_rng = random.Random(hour_rng_seed)
    episode_rewards: List[float] = []
    episode_probes: List[int] = []
    losses: List[float] = []
    total_steps = 0

    ep_iter = range(n_episodes)
    if not quiet:
        ep_iter = tqdm(ep_iter, desc=f"train_scoring_{variant}")

    for _ep in ep_iter:
        pair = pair_rng.choice(pair_pool)
        start_hour = hour_rng.choice(TRAINING_HOURS)
        env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=start_hour)
        state = env.observe_scoring()
        ep_reward = 0.0
        for _ in range(episode_length):
            if state["paths"].shape[0] == 0:
                break
            action = int(agent.act(state))
            if action >= len(env.available_paths):
                action = 0
            reward, done, _info = env.apply_action(action, probe="full")
            ep_reward += reward
            next_state = env.observe_scoring()
            agent.remember(state, action, reward, next_state, done)
            total_steps += 1
            if total_steps % grad_every == 0:
                loss = agent.replay()
                if loss is not None:
                    losses.append(float(loss))
            state = next_state
            if done:
                break
        agent.episodes += 1
        agent.epsilon = max(hp.epsilon_end, agent.epsilon * hp.epsilon_decay)
        episode_rewards.append(ep_reward / max(1, episode_length))
        episode_probes.append(int(env.num_latency_probes + env.num_bandwidth_probes))

    ckpt = checkpoint_path or (
        run_path
        / (
            "dqn_scoring_simple_model.pth"
            if variant == "simple"
            else "dqn_scoring_enhanced_model.pth"
        )
    )
    payload: Dict[str, Any] = {
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
        "steps": agent.steps,
        "episodes": agent.episodes,
        "global_dim": SCORING_GLOBAL_DIM,
        "path_dim": PATH_FEATURE_DIM,
        "hidden_dim": hp.hidden_dim,
        "hyperparams": asdict(hp),
        "goodput_cap_mbps": goodput_cap,
        "reward_weights": reward_weights.to_dict(),
        "pair_pool": [list(p) for p in pair_pool],
        "variant": variant,
        "optimizer": agent.optimizer.state_dict(),
    }
    if variant == "enhanced":
        payload["scheduler"] = agent.scheduler.state_dict()
        payload["beta"] = agent.beta
        payload["config"] = config

    torch.save(payload, ckpt)

    stats = {
        "variant": variant,
        "num_episodes": n_episodes,
        "episode_length_hours": episode_length,
        "episode_rewards": episode_rewards,
        "episode_probes": episode_probes,
        "losses": losses,
        "final_epsilon": agent.epsilon,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_probes_per_episode": (
            float(np.mean(episode_probes)) if episode_probes else 0.0
        ),
        "reward_weights": reward_weights.to_dict(),
        "goodput_cap_mbps": goodput_cap,
        "pair_pool_size": len(pair_pool),
        "hyperparams": asdict(hp),
        "checkpoint": str(ckpt),
    }
    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def train_conditional_scoring_dqn(
    run_path: Path,
    hp: Optional[ScoringHyperparams] = None,
    *,
    num_episodes: Optional[int] = None,
    episode_length: int = EPISODE_LENGTH,
    checkpoint_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    env_seed: int = 42,
    pair_rng_seed: int = 123,
    hour_rng_seed: int = 456,
    weight_rng_seed: int = 789,
    quiet: bool = False,
    run_context: Optional[Tuple] = None,
    architecture: str = CONDITIONAL_ARCH_WEIGHT_FILM,
) -> Dict[str, Any]:
    """Train a path-scoring DQN with reward weights in the global state.

    Each episode samples a :class:`~src.rl.reward_profiles.RewardProfile` so the
    policy learns to optimize different composite objectives at inference time.

    ``architecture`` selects how the reward-weight vector conditions the network:
    ``CONDITIONAL_ARCH_WEIGHT_FILM`` (default) uses FiLM modulation of per-path
    features; ``CONDITIONAL_ARCH_VALUE_CONCAT`` concatenates the weights into the
    value stream only (the naive ablation expected not to re-rank paths);
    ``CONDITIONAL_ARCH_LEGACY`` uses plain dueling-concat with the full global
    vector in both streams.
    """
    valid_archs = (
        CONDITIONAL_ARCH_WEIGHT_FILM,
        CONDITIONAL_ARCH_VALUE_CONCAT,
        CONDITIONAL_ARCH_LEGACY,
    )
    if architecture not in valid_archs:
        raise ValueError(
            f"Unknown conditional architecture {architecture!r}; "
            f"expected one of {valid_archs}"
        )
    hp = hp or ScoringHyperparams.from_env()
    if run_context is None:
        run_context = load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, goodput_cap = run_context
    env = make_env(
        topology_data,
        path_store,
        link_states,
        pair_pool,
        episode_length=episode_length,
        rng_seed=env_seed,
        reward_weights=RewardWeights(),
    )

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
        use_batch_norm=False,
        use_prioritized_replay=hp.use_prioritized_replay,
        use_dueling_dqn=True,
        use_double_dqn=True,
        tau=hp.tau,
    )
    set_conditional_weight_encoding("policy")
    if architecture == CONDITIONAL_ARCH_WEIGHT_FILM:
        agent = ConditionalPathScoringDQNAgent(
            global_dim=CONDITIONAL_SCORING_GLOBAL_DIM,
            path_dim=PATH_FEATURE_DIM,
            config=config,
        )
    elif architecture == CONDITIONAL_ARCH_VALUE_CONCAT:
        agent = ValueConcatConditionalPathScoringDQNAgent(
            global_dim=CONDITIONAL_SCORING_GLOBAL_DIM,
            path_dim=PATH_FEATURE_DIM,
            config=config,
        )
    else:
        agent = LegacyConditionalPathScoringDQNAgent(
            global_dim=CONDITIONAL_SCORING_GLOBAL_DIM,
            path_dim=PATH_FEATURE_DIM,
            config=config,
        )

    base_episodes = num_episodes or _target_episodes(len(pair_pool), episode_length)
    n_episodes = max(50, int(base_episodes * conditional_episode_multiplier()))
    grad_every = _gradient_every()
    pair_rng = random.Random(pair_rng_seed)
    hour_rng = random.Random(hour_rng_seed)
    weight_rng = random.Random(weight_rng_seed)
    episode_rewards: List[float] = []
    episode_probes: List[int] = []
    episode_weight_names: List[str] = []
    losses: List[float] = []
    total_steps = 0

    profile_list = get_conditional_training_profiles()
    profile_schedule = stratified_training_profile_schedule(
        n_episodes, profile_list, weight_rng
    )
    ep_iter = enumerate(profile_schedule)
    if not quiet:
        ep_iter = tqdm(ep_iter, total=n_episodes, desc="train_conditional_scoring")

    for _ep, sampled in ep_iter:
        env.reward_weights = sampled.weights
        episode_weight_names.append(sampled.name)

        pair = pair_rng.choice(pair_pool)
        start_hour = hour_rng.choice(TRAINING_HOURS)
        env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=start_hour)
        state = env.observe_scoring_conditional()
        ep_reward = 0.0

        for _ in range(episode_length):
            if state["paths"].shape[0] == 0:
                break
            action = int(agent.act(state))
            if action >= len(env.available_paths):
                action = 0
            reward, done, _info = env.apply_action(action, probe="full")
            ep_reward += reward
            next_state = env.observe_scoring_conditional()
            agent.remember(state, action, reward, next_state, done)
            total_steps += 1
            if total_steps % grad_every == 0:
                loss = agent.replay()
                if loss is not None:
                    losses.append(float(loss))
            state = next_state
            if done:
                break

        agent.episodes += 1
        agent.epsilon = max(hp.epsilon_end, agent.epsilon * hp.epsilon_decay)
        episode_rewards.append(ep_reward / max(1, episode_length))
        episode_probes.append(int(env.num_latency_probes + env.num_bandwidth_probes))

    _ckpt_names = {
        CONDITIONAL_ARCH_WEIGHT_FILM: "dqn_conditional_scoring_model.pth",
        CONDITIONAL_ARCH_VALUE_CONCAT: "dqn_conditional_value_concat_model.pth",
        CONDITIONAL_ARCH_LEGACY: "dqn_conditional_concat_model.pth",
    }
    default_ckpt_name = _ckpt_names[architecture]
    ckpt = checkpoint_path or (run_path / default_ckpt_name)
    payload: Dict[str, Any] = {
        "model_type": "conditional_scoring_dqn",
        "architecture": architecture,
        "weight_encoding": get_conditional_weight_encoding(),
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
        "steps": agent.steps,
        "episodes": agent.episodes,
        "global_dim": CONDITIONAL_SCORING_GLOBAL_DIM,
        "path_dim": PATH_FEATURE_DIM,
        "scoring_global_dim": SCORING_GLOBAL_DIM,
        "weight_dim": REWARD_WEIGHT_DIM,
        "hyperparams": asdict(hp),
        "goodput_cap_mbps": goodput_cap,
        "training_profiles": [p.to_dict() for p in profile_list],
        "pair_pool": [list(p) for p in pair_pool],
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict(),
        "beta": agent.beta,
        "config": config,
    }
    torch.save(payload, ckpt)

    stats = {
        "model_type": "conditional_scoring_dqn",
        "architecture": architecture,
        "weight_encoding": get_conditional_weight_encoding(),
        "num_episodes": n_episodes,
        "base_episodes_before_multiplier": base_episodes,
        "episode_length_hours": episode_length,
        "episode_rewards": episode_rewards,
        "episode_probes": episode_probes,
        "episode_weight_profiles": episode_weight_names,
        "losses": losses,
        "final_epsilon": agent.epsilon,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_probes_per_episode": (
            float(np.mean(episode_probes)) if episode_probes else 0.0
        ),
        "goodput_cap_mbps": goodput_cap,
        "pair_pool_size": len(pair_pool),
        "global_dim": CONDITIONAL_SCORING_GLOBAL_DIM,
        "training_profiles": [p.name for p in profile_list],
        "hyperparams": asdict(hp),
        "checkpoint": str(ckpt),
    }
    _stats_names = {
        CONDITIONAL_ARCH_WEIGHT_FILM: "dqn_conditional_training_stats.json",
        CONDITIONAL_ARCH_VALUE_CONCAT: (
            "dqn_conditional_value_concat_training_stats.json"
        ),
        CONDITIONAL_ARCH_LEGACY: "dqn_conditional_concat_training_stats.json",
    }
    default_stats_name = _stats_names[architecture]
    out_stats = stats_path or (run_path / default_stats_name)
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    return stats
