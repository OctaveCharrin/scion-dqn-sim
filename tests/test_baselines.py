"""Tests for the baseline path selectors.

These tests use lightweight stub path objects exposing the subset of the
``SCIONPath`` API the selectors actually read (``as_sequence``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.baselines.ecmp import ECMPSelector
from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.random_selection import RandomSelector
from src.baselines.scion_default import SCIONDefaultSelector
from src.baselines.shortest_path import ShortestPathSelector
from src.baselines.widest_path import WidestPathSelector


@dataclass
class _StubPath:
    as_sequence: Tuple[int, ...] = field(default_factory=tuple)


def _make_paths(hops_per_path: List[int]) -> List[_StubPath]:
    return [_StubPath(as_sequence=tuple(range(h + 1))) for h in hops_per_path]


def _metrics(latencies=None, bandwidths=None, losses=None):
    n = max(len(latencies or []), len(bandwidths or []), len(losses or []))
    out = []
    for i in range(n):
        m = {}
        if latencies is not None:
            m["latency_ms"] = latencies[i]
        if bandwidths is not None:
            m["bandwidth_mbps"] = bandwidths[i]
        if losses is not None:
            m["loss_rate"] = losses[i]
        out.append(m)
    return out


FLOW = {"src": 1, "dst": 42}
STATE = np.zeros(1, dtype=np.float32)


def test_shortest_path_picks_fewest_hops():
    paths = _make_paths([4, 2, 3])
    idx = ShortestPathSelector().select_path(paths, _metrics(latencies=[10, 10, 10]), FLOW, STATE)
    assert idx == 1


def test_widest_path_picks_max_bandwidth():
    paths = _make_paths([3, 3, 3])
    idx = WidestPathSelector().select_path(
        paths, _metrics(bandwidths=[100, 500, 250]), FLOW, STATE
    )
    assert idx == 1


def test_lowest_latency_picks_min_latency():
    paths = _make_paths([3, 3, 3])
    idx = LowestLatencySelector().select_path(
        paths, _metrics(latencies=[50, 20, 30]), FLOW, STATE
    )
    assert idx == 1


def test_ecmp_picks_among_shortest_paths_deterministically():
    paths = _make_paths([4, 2, 2, 3])
    sel = ECMPSelector()
    idx_a = sel.select_path(paths, _metrics(latencies=[10, 10, 10, 10]), FLOW, STATE)
    idx_b = sel.select_path(paths, _metrics(latencies=[10, 10, 10, 10]), FLOW, STATE)
    # Same flow key must be stable and choose one of the two shortest paths.
    assert idx_a == idx_b
    assert idx_a in {1, 2}


def test_random_selector_returns_valid_index():
    paths = _make_paths([2, 3, 4])
    idx = RandomSelector().select_path(paths, _metrics(latencies=[10, 20, 30]), FLOW, STATE)
    assert 0 <= idx < len(paths)


def test_scion_default_prefers_shortest_then_lowest_latency():
    paths = _make_paths([4, 2, 2, 3])
    idx = SCIONDefaultSelector().select_path(
        paths, _metrics(latencies=[5, 30, 15, 10]), FLOW, STATE
    )
    # Shortest are indices 1 and 2; among those, index 2 has lower latency.
    assert idx == 2


def test_selectors_handle_empty_paths_gracefully():
    for selector in (
        ShortestPathSelector(),
        WidestPathSelector(),
        LowestLatencySelector(),
        ECMPSelector(),
        RandomSelector(),
        SCIONDefaultSelector(),
    ):
        assert selector.select_path([], [], FLOW, STATE) == 0
