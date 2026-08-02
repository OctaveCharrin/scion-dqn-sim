"""Tests for the across-seed aggregation.

The aggregation is what turns five per-seed evaluations into the intervals the
study reports, so the two things worth pinning down are that a group's rows are
matched across seeds by key rather than by position, and that the per-seed values
kept alongside the interval are the values the interval was computed from -- the
figures draw their traces from that column, so a mismatch would put a picture and
a number out of step without either looking wrong on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

analyze_seed_results = pytest.importorskip("analyze_seed_results")
aggregate_rows = analyze_seed_results.aggregate_rows


def _rows(values):
    return [{"method": "a", "n_paths": "2", "regret_mean": str(v)} for v in values]


def test_aggregate_mean_and_interval_over_seeds() -> None:
    per_seed = {f"seed{i+1}": _rows([v]) for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])}
    (row,) = aggregate_rows(per_seed, ["method", "n_paths"], ["regret_mean"])

    assert row["n_seeds"] == 5
    assert row["regret_mean_mean"] == pytest.approx(3.0)
    assert row["regret_mean_sd"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    # Symmetric t-interval, and wide enough to contain the mean.
    assert row["regret_mean_ci_lo"] < 3.0 < row["regret_mean_ci_hi"]
    assert (row["regret_mean_ci_hi"] - 3.0) == pytest.approx(3.0 - row["regret_mean_ci_lo"])


def test_seedvals_column_matches_the_aggregated_values() -> None:
    values = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    per_seed = {f"seed{i+1}": _rows([v]) for i, v in enumerate(values)}
    (row,) = aggregate_rows(per_seed, ["method", "n_paths"], ["regret_mean"])

    kept = [float(v) for v in row["regret_mean_seedvals"].split(";")]
    assert kept == pytest.approx(sorted(values, reverse=True))
    assert np.mean(kept) == pytest.approx(row["regret_mean_mean"])


def test_rows_are_matched_by_key_not_by_position() -> None:
    """Seeds may emit their rows in any order; grouping must follow the key."""
    per_seed = {
        "seed1": [
            {"method": "a", "n_paths": "2", "regret_mean": "1.0"},
            {"method": "b", "n_paths": "2", "regret_mean": "10.0"},
        ],
        "seed2": [
            {"method": "b", "n_paths": "2", "regret_mean": "12.0"},
            {"method": "a", "n_paths": "2", "regret_mean": "3.0"},
        ],
    }
    out = aggregate_rows(per_seed, ["method", "n_paths"], ["regret_mean"])
    rows = {r["method"]: r for r in out}

    assert rows["a"]["regret_mean_mean"] == pytest.approx(2.0)
    assert rows["b"]["regret_mean_mean"] == pytest.approx(11.0)


def test_passthrough_columns_are_carried_but_not_aggregated() -> None:
    per_seed = {
        f"seed{i}": [
            {"method": "a", "n_paths": "2", "method_label": "A", "regret_mean": "1.0"}
        ]
        for i in (1, 2)
    }
    (row,) = aggregate_rows(
        per_seed, ["method"], ["regret_mean"], passthrough=["method_label"]
    )
    assert row["method_label"] == "A"
    assert "method_label_mean" not in row
