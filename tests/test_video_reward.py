"""Tests for the video env reward function."""

from __future__ import annotations

from src.ns3env.reward import RewardWeights, compute_reward


def test_reward_bounds():
    r = compute_reward(throughput_mbps=100.0, rtt_ms=0.0, loss=0.0, cap_mbps=10.0)
    assert r == 1.0  # goodput saturates at cap, trust perfect
    r = compute_reward(throughput_mbps=0.0, rtt_ms=1000.0, loss=1.0, cap_mbps=10.0)
    assert -1.0 <= r <= 1.0


def test_higher_throughput_higher_reward():
    lo = compute_reward(throughput_mbps=2.0, rtt_ms=20.0, loss=0.0, cap_mbps=10.0)
    hi = compute_reward(throughput_mbps=8.0, rtt_ms=20.0, loss=0.0, cap_mbps=10.0)
    assert hi > lo


def test_loss_and_delay_penalize_trust():
    clean = compute_reward(throughput_mbps=5.0, rtt_ms=10.0, loss=0.0, cap_mbps=10.0)
    lossy = compute_reward(throughput_mbps=5.0, rtt_ms=10.0, loss=0.5, cap_mbps=10.0)
    slow = compute_reward(throughput_mbps=5.0, rtt_ms=180.0, loss=0.0, cap_mbps=10.0)
    assert clean > lossy
    assert clean > slow


def test_weights_flip_path_ranking():
    # Weights should be able to flip which path is preferred.
    tput = RewardWeights(w1=1.0, w2=0.0)
    trust = RewardWeights(w1=0.1, w2=0.9, w3=1.0, w4=1.0)
    fast_lossy = dict(throughput_mbps=9.0, rtt_ms=120.0, loss=0.4, cap_mbps=10.0)
    slow_clean = dict(throughput_mbps=3.0, rtt_ms=20.0, loss=0.0, cap_mbps=10.0)
    # Throughput-only prefers the fast (lossy) path...
    assert compute_reward(**fast_lossy, weights=tput) > compute_reward(
        **slow_clean, weights=tput
    )
    # ...trust-heavy weights prefer the clean (slow) path.
    assert compute_reward(**slow_clean, weights=trust) > compute_reward(
        **fast_lossy, weights=trust
    )


def test_zero_cap_safe():
    r = compute_reward(throughput_mbps=1.0, rtt_ms=10.0, loss=0.0, cap_mbps=0.0)
    assert -1.0 <= r <= 1.0
