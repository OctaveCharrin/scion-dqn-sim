"""
Smoke test for the multipath video DQN training loop (mock data plane).

This is the Phase-1 milestone in code form: on a deliberately asymmetric,
time-varying scenario the agent should (a) improve over training and (b) beat a
state-blind round-robin scheduler and a fixed single-path policy on goodput.

Kept small so it runs on CPU/Windows in a few seconds. Assertions are lenient to
avoid RNG flakiness while still catching a broken learning loop.
"""

from __future__ import annotations

from src.baselines.multipath_baselines import RoundRobinSelector, StaticPathSelector
from src.ns3env.dataplane import MockTraceConfig, MockTraceDataPlane
from src.rl.video_mpquic_train import (
    VideoTrainHyperparams,
    evaluate_policy,
    train_video_dqn,
)


def _factory():
    # Strongly asymmetric: path 0 is usually best but each path peaks at a
    # different phase, so a static or round-robin policy leaves goodput on the table.
    return MockTraceDataPlane(
        MockTraceConfig(
            num_paths=3,
            episode_segments=16,
            base_mbps=(8.0, 4.0, 2.0),
            amp=(0.6, 0.6, 0.6),
            period_s=(30.0, 30.0, 30.0),
        )
    )


def test_training_improves_and_beats_baselines():
    hp = VideoTrainHyperparams(
        hidden_dim=64,
        batch_size=64,
        min_buffer_size=200,
        epsilon_decay=0.97,
    )
    stats, agent = train_video_dqn(
        dataplane_factory=_factory,
        num_episodes=180,
        hp=hp,
        seed=0,
        quiet=True,
    )

    rewards = stats["episode_rewards"]
    head = sum(rewards[:20]) / 20
    tail = sum(rewards[-20:]) / 20
    assert tail > head, f"no learning: head={head:.3f} tail={tail:.3f}"

    agent_eval = evaluate_policy(
        agent, dataplane_factory=_factory, episodes=20, is_agent=True
    )
    rr_eval = evaluate_policy(
        RoundRobinSelector(), dataplane_factory=_factory, episodes=20
    )
    static_eval = evaluate_policy(
        StaticPathSelector(2), dataplane_factory=_factory, episodes=20
    )

    # Beat the state-blind scheduler and the worst static path on goodput.
    assert agent_eval["mean_goodput_mbps"] > rr_eval["mean_goodput_mbps"]
    assert agent_eval["mean_goodput_mbps"] > static_eval["mean_goodput_mbps"]
