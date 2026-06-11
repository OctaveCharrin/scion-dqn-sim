"""Smoke test for the ABR train+compare driver (mock backend, tiny run)."""

from __future__ import annotations

import json

from src.baselines.abr_baselines import ABR_SELECTORS
from src.rl.video_abr_train import run_comparison


def test_abr_run_comparison_structure_and_output(tmp_path):
    out = tmp_path / "abr_results.json"
    results = run_comparison(
        backend="mock",
        scenario="varying",
        num_episodes=3,
        eval_episodes=2,
        seed=0,
        out_path=out,
        quiet=True,
    )

    expected = {"dqn", *ABR_SELECTORS}
    assert set(results["methods"]) == expected
    for m in results["methods"].values():
        assert isinstance(m["mean_reward"], float)
        assert isinstance(m["mean_vmaf"], float)
        assert isinstance(m["mean_rebuffer_s_per_ep"], float)

    assert out.exists()
    with open(out) as f:
        json.load(f)
