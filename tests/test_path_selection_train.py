"""Tests for DQN training schedule helpers."""

from __future__ import annotations

from src.rl.path_selection_train import (
    EPISODE_LENGTH,
    _gradient_every,
    _target_episodes,
    _training_pair_cap,
)


def test_training_pair_cap_default():
    assert _training_pair_cap(2450) == 64
    assert _training_pair_cap(10) == 10


def test_training_pair_cap_env(monkeypatch):
    monkeypatch.setenv("DQN_TRAIN_PAIR_CAP", "32")
    assert _training_pair_cap(2450) == 32


def test_target_episodes_scales_with_capped_pool():
    # 200 * 64 = 12800 steps -> 533 episodes at length 24
    n = _target_episodes(2450)
    assert n == max(50, min(20_000, 200 * 64) // EPISODE_LENGTH)


def test_target_episodes_env_override(monkeypatch):
    monkeypatch.setenv("DQN_TRAIN_EPISODES", "100")
    assert _target_episodes(2450) == 100


def test_gradient_every_default():
    assert _gradient_every() == 4


def test_gradient_every_env(monkeypatch):
    monkeypatch.setenv("DQN_GRADIENT_EVERY", "1")
    assert _gradient_every() == 1
