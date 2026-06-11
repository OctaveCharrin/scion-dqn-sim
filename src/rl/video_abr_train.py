"""
Train + compare adaptive-bitrate (ABR) policies for the Phase-2 video QoE env.

Reuses the path-scoring DQN unchanged -- the env presents each bitrate option as a
scoring candidate, so ``EnhancedPathScoringDQNAgent`` picks a rung per segment.
This is the single-path ABR validation (canonical Pensieve setup): the playback
buffer makes rebuffering a delayed consequence of earlier choices, so a learned
policy can beat reactive rate-/buffer-based heuristics.

CLI::

    uv run python -m src.rl.video_abr_train --episodes 300 --eval-episodes 30
    NS3_DIR=~/ns-3-dev uv run python -m src.rl.video_abr_train --backend ns3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.baselines.abr_baselines import ABR_SELECTORS
from src.ns3env.abr import QoEWeights
from src.ns3env.abr_env import VideoAbrEnv
from src.ns3env.dataplane import DataPlane, MockTraceConfig, MockTraceDataPlane
from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
from src.rl.video_mpquic_train import VideoTrainHyperparams

EnvFactory = Callable[[], VideoAbrEnv]

# Mock scenarios: ``varying`` gives a single path whose capacity swings widely, so
# a fixed bitrate either stalls (too high) or wastes quality (too low) and an
# adaptive policy wins. ``default`` is the gentler Phase-1 path-0 profile.
_SCENARIOS: Dict[str, MockTraceConfig] = {
    "default": MockTraceConfig(num_paths=1),
    "varying": MockTraceConfig(
        num_paths=1,
        base_mbps=(2.5,),
        amp=(0.8,),
        period_s=(20.0,),
        base_rtt_ms=(30.0,),
        noise_std=0.08,
    ),
}


def _make_dataplane(backend: str, scenario: str) -> DataPlane:
    if backend == "mock":
        cfg = _SCENARIOS.get(scenario)
        if cfg is None:
            raise ValueError(f"unknown scenario {scenario!r} ({list(_SCENARIOS)})")
        return MockTraceDataPlane(cfg)
    if backend == "ns3":
        from src.ns3env.dataplane import Ns3Config, Ns3DataPlane

        return Ns3DataPlane(num_paths=1, config=Ns3Config())
    raise ValueError(f"unknown backend {backend!r} (use 'mock' or 'ns3')")


def _make_agent(
    env: VideoAbrEnv, hp: VideoTrainHyperparams
) -> EnhancedPathScoringDQNAgent:
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
        use_prioritized_replay=True,
        use_dueling_dqn=True,
        use_double_dqn=True,
        tau=hp.tau,
    )
    return EnhancedPathScoringDQNAgent(
        global_dim=env.global_dim, path_dim=env.path_dim, config=config
    )


def train_abr_dqn(
    *,
    env_factory: EnvFactory,
    num_episodes: int = 300,
    hp: Optional[VideoTrainHyperparams] = None,
    seed: int = 42,
    quiet: bool = False,
) -> tuple[Dict[str, Any], EnhancedPathScoringDQNAgent]:
    """Train the scoring DQN on the ABR env; return ``(stats, agent)``."""
    hp = hp or VideoTrainHyperparams()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = env_factory()
    agent = _make_agent(env, hp)

    ep_iter: Any = range(num_episodes)
    if not quiet:
        try:
            from tqdm import tqdm

            ep_iter = tqdm(ep_iter, desc="train_abr_dqn")
        except ImportError:
            pass

    episode_rewards: List[float] = []
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
                agent.replay()
            state = next_state
            if done:
                break
        agent.episodes += 1
        agent.epsilon = max(hp.epsilon_end, agent.epsilon * hp.epsilon_decay)
        episode_rewards.append(ep_reward / max(1, n_steps))

    stats = {
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_reward_last_decile": _tail_mean(episode_rewards),
        "final_epsilon": agent.epsilon,
    }
    return stats, agent


def _tail_mean(values: List[float], frac: float = 0.1) -> float:
    if not values:
        return 0.0
    k = max(1, int(len(values) * frac))
    return float(np.mean(values[-k:]))


def evaluate_abr_policy(
    selector: Any,
    *,
    env_factory: EnvFactory,
    episodes: int = 30,
    seed: int = 10_000,
    is_agent: bool = False,
) -> Dict[str, float]:
    """Roll out a policy; report mean reward, mean bitrate and rebuffering."""
    env = env_factory()
    rewards: List[float] = []
    bitrates: List[float] = []
    vmafs: List[float] = []
    rebuffer_per_ep: List[float] = []
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        if hasattr(selector, "reset"):
            selector.reset()
        ep_r: List[float] = []
        ep_b: List[float] = []
        ep_v: List[float] = []
        ep_rebuf = 0.0
        while True:
            if is_agent:
                action = int(selector.act(state, evaluate=True))
            else:
                action = int(selector.select(state))
            state, reward, done, info = env.step(action)
            ep_r.append(reward)
            ep_b.append(info["chosen_bitrate_mbps"])
            ep_v.append(info["chosen_vmaf"])
            ep_rebuf += info["rebuffer_s"]
            if done:
                break
        rewards.append(float(np.mean(ep_r)))
        bitrates.append(float(np.mean(ep_b)))
        vmafs.append(float(np.mean(ep_v)))
        rebuffer_per_ep.append(ep_rebuf)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_vmaf": float(np.mean(vmafs)),
        "mean_bitrate_mbps": float(np.mean(bitrates)),
        "mean_rebuffer_s_per_ep": float(np.mean(rebuffer_per_ep)),
        "episodes": episodes,
    }


def run_comparison(
    *,
    backend: str = "mock",
    scenario: str = "varying",
    num_episodes: int = 300,
    eval_episodes: int = 30,
    seed: int = 42,
    eval_seed: int = 10_000,
    qoe_weights: Optional[QoEWeights] = None,
    out_path: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Train the ABR DQN and compare it against all ABR heuristics."""
    qoe_weights = qoe_weights or QoEWeights()
    dataplane = _make_dataplane(backend, scenario)
    env_factory: EnvFactory = lambda: VideoAbrEnv(dataplane, qoe_weights=qoe_weights)

    train_stats, agent = train_abr_dqn(
        env_factory=env_factory, num_episodes=num_episodes, seed=seed, quiet=quiet
    )

    methods: Dict[str, Dict[str, float]] = {}
    methods["dqn"] = evaluate_abr_policy(
        agent,
        env_factory=env_factory,
        episodes=eval_episodes,
        seed=eval_seed,
        is_agent=True,
    )
    for name, selector_cls in ABR_SELECTORS.items():
        methods[name] = evaluate_abr_policy(
            selector_cls(),
            env_factory=env_factory,
            episodes=eval_episodes,
            seed=eval_seed,
            is_agent=False,
        )

    best_baseline = max(
        (n for n in methods if n != "dqn"),
        key=lambda n: methods[n]["mean_reward"],
    )
    dqn_r = methods["dqn"]["mean_reward"]
    base_r = methods[best_baseline]["mean_reward"]

    results: Dict[str, Any] = {
        "backend": backend,
        "scenario": scenario,
        "num_train_episodes": num_episodes,
        "eval_episodes": eval_episodes,
        "train_avg_reward_last_decile": train_stats.get("avg_reward_last_decile"),
        "methods": methods,
        "best_baseline": best_baseline,
        "dqn_minus_best_baseline_reward": dqn_r - base_r,
        "dqn_beats_best_baseline": dqn_r > base_r,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        results["out_path"] = str(out_path)

    if backend == "ns3" and hasattr(dataplane, "close"):
        dataplane.close()

    return results


def _print_table(results: Dict[str, Any]) -> None:
    methods = results["methods"]
    order = sorted(methods, key=lambda n: methods[n]["mean_reward"], reverse=True)
    width = max(len(n) for n in methods)
    print(
        f"\nABR  backend: {results['backend']}  scenario: {results.get('scenario')}  "
        f"(train {results['num_train_episodes']} ep, eval {results['eval_episodes']} ep)"
    )
    print(
        f"{'method':<{width}}  {'reward':>9}  {'vmaf':>7}  "
        f"{'bitrate_mbps':>13}  {'rebuf_s/ep':>11}"
    )
    print("-" * (width + 48))
    for name in order:
        m = methods[name]
        star = "  <- DQN" if name == "dqn" else ""
        print(
            f"{name:<{width}}  {m['mean_reward']:>9.4f}  {m['mean_vmaf']:>7.2f}  "
            f"{m['mean_bitrate_mbps']:>13.3f}  {m['mean_rebuffer_s_per_ep']:>11.2f}{star}"
        )
    verdict = "BEATS" if results["dqn_beats_best_baseline"] else "does NOT beat"
    print(
        f"\nDQN {verdict} best baseline '{results['best_baseline']}' "
        f"by {results['dqn_minus_best_baseline_reward']:+.4f} reward."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["mock", "ns3"], default="mock")
    p.add_argument("--scenario", choices=list(_SCENARIOS), default="varying")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    results = run_comparison(
        backend=args.backend,
        scenario=args.scenario,
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        out_path=args.out,
        quiet=args.quiet,
    )
    _print_table(results)


if __name__ == "__main__":
    main()
