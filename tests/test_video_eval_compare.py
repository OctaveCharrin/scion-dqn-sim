"""Smoke test for the Phase-1 train+compare driver (mock backend, tiny run)."""

from __future__ import annotations

import json

from src.baselines.multipath_baselines import MULTIPATH_SELECTORS
from src.rl.video_eval_compare import run_comparison


def test_run_comparison_structure_and_output(tmp_path):
    out = tmp_path / "results.json"
    results = run_comparison(
        backend="mock",
        scenario="crossover",
        num_episodes=3,
        eval_episodes=2,
        seed=0,
        out_path=out,
        quiet=True,
    )

    # Every method (DQN + all registered baselines) is reported with both metrics.
    expected = {"dqn", *MULTIPATH_SELECTORS}
    assert set(results["methods"]) == expected
    for m in results["methods"].values():
        assert isinstance(m["mean_reward"], float)
        assert isinstance(m["mean_goodput_mbps"], float)

    assert results["best_baseline"] in MULTIPATH_SELECTORS
    assert isinstance(results["dqn_beats_best_baseline"], bool)

    # Results were persisted and round-trip as JSON.
    assert out.exists()
    with open(out) as f:
        json.load(f)
