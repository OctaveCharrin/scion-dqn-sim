"""Tests for evaluation figure helpers."""

from __future__ import annotations

from src.pipeline.figures import (
    figure2_filename_for_profile,
    generate_figure2_for_profiles,
    summaries_for_profile,
)


def test_figure2_filenames() -> None:
    assert figure2_filename_for_profile() == "figure2_path_reward.png"
    assert figure2_filename_for_profile("balanced") == "figure2_path_reward.png"
    assert figure2_filename_for_profile("throughput") == "figure2_path_reward_throughput.png"


def test_summaries_for_profile() -> None:
    data = {
        "profiles": ["balanced", "throughput"],
        "results": [
            {"method": "conditional_dqn", "profile": "balanced", "reward_mean": 0.9, "reward_std": 0.1, "n_selections": 100},
            {"method": "widest_path", "profile": "balanced", "reward_mean": 0.85, "reward_std": 0.12, "n_selections": 100},
            {"method": "conditional_dqn", "profile": "throughput", "reward_mean": 0.95, "reward_std": 0.05, "n_selections": 100},
        ],
    }
    balanced = summaries_for_profile(data, "balanced")
    assert set(balanced) == {"conditional_dqn", "widest_path"}
    assert balanced["conditional_dqn"]["reward_mean"] == 0.9


def test_generate_figure2_for_profiles(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")

    from matplotlib import pyplot as plt

    plt.switch_backend("Agg")
    data = {
        "profiles": ["balanced", "latency"],
        "methods": ["conditional_dqn", "widest_path"],
        "results": [
            {"method": "conditional_dqn", "profile": "balanced", "reward_mean": 0.9, "reward_std": 0.1, "n_selections": 50},
            {"method": "widest_path", "profile": "balanced", "reward_mean": 0.85, "reward_std": 0.12, "n_selections": 50},
            {"method": "conditional_dqn", "profile": "latency", "reward_mean": 0.8, "reward_std": 0.15, "n_selections": 50},
            {"method": "widest_path", "profile": "latency", "reward_mean": 0.82, "reward_std": 0.11, "n_selections": 50},
        ],
    }
    paths = generate_figure2_for_profiles(tmp_path, data)
    assert len(paths) == 2
    assert (tmp_path / "figure2_path_reward.png").is_file()
    assert (tmp_path / "figure2_path_reward_latency.png").is_file()
