"""Tests for ``EvaluationPathSelectionEnv`` (path-selection RL env).

Covers:

* Probe semantics: ``probe_path_*`` returns the *measured* latency without
  the probe overhead added in; the cost is reported separately on
  ``last_probe_cost_ms`` / ``total_probe_cost_ms``.
* Stateful episodes: ``step`` advances ``hour_idx`` and signals ``done``
  after ``episode_length`` steps.
* Per-pair link state lookup: when ``link_states`` uses the multi-pair
  ``by_pair`` schema, the env reads the right pair's path metrics.
* Action mask aligns with the current pair's path count.
"""

from __future__ import annotations

import numpy as np

from src.simulation.evaluation_env import EvaluationPathSelectionEnv
from src.simulation.path_store import InMemoryPathStore


def _make_paths(latencies):
    return [
        {
            "hops": [{"as": 1}, {"as": 2}, {"as": 3}],
            "static_metrics": {
                "hop_count": 2,
                "total_latency": float(lat),
                "min_bandwidth": 1000.0,
            },
        }
        for lat in latencies
    ]


def _path_store(pair_to_latencies):
    store = InMemoryPathStore()
    for pair, lats in pair_to_latencies.items():
        store.set_paths(pair[0], pair[1], _make_paths(lats))
    return store


def _link_states_with_pair(pair_states):
    """Build link_states[hour]['by_pair']['pair_x_y']['path_i'] = state."""
    out = {}
    for hour, blocks in pair_states.items():
        out[hour] = {"by_pair": {f"pair_{p[0]}_{p[1]}": st for p, st in blocks.items()}}
    return out


def test_probe_path_latency_does_not_inflate_returned_latency():
    store = _path_store({(1, 3): [25.0, 40.0]})
    link_states = _link_states_with_pair(
        {0: {(1, 3): {"path_0": {"latency_ms": 25.0, "loss_rate": 0.0, "utilization": 0.1}}}}
    )
    env = EvaluationPathSelectionEnv(
        topology_data={},
        path_store=store,
        link_states=link_states,
        latency_probe_cost_ms=10.0,
        bandwidth_probe_cost_ms=100.0,
        per_hop_probe_cost_ms=0.5,
        per_hop_full_probe_cost_ms=20.0,
        pair_pool=[(1, 3)],
    )
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    out = env.probe_path_latency(0)
    assert out["latency_ms"] == 25.0  # NOT 25 + 10
    # Probe cost = 10 (base) + 0.5 * hops (2) = 11.0
    assert out["probe_cost_ms"] == 11.0
    assert env.last_probe_cost_ms == 11.0
    assert env.total_probe_cost_ms == 11.0


def test_probe_path_full_returns_dynamic_bandwidth():
    store = _path_store({(1, 3): [25.0]})
    link_states = _link_states_with_pair(
        {
            0: {
                (1, 3): {
                    "path_0": {
                        "latency_ms": 30.0,
                        "available_bandwidth_mbps": 250.0,
                        "loss_rate": 0.05,
                        "utilization": 0.4,
                    }
                }
            }
        }
    )
    env = EvaluationPathSelectionEnv(
        topology_data={},
        path_store=store,
        link_states=link_states,
        pair_pool=[(1, 3)],
    )
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    out = env.probe_path_full(0)
    assert out["bandwidth_mbps"] == 250.0
    assert out["loss_rate"] == 0.05
    # Full probe billed: base 100 + 20 * hops (2) = 140.
    assert out["probe_cost_ms"] == 140.0


def test_step_advances_hour_and_terminates_on_episode_length():
    store = _path_store({(1, 3): [25.0]})
    link_states = {
        h: {"by_pair": {"pair_1_3": {"path_0": {"latency_ms": 25.0 + h, "utilization": 0.0, "loss_rate": 0.0, "available_bandwidth_mbps": 1000.0}}}}
        for h in range(5)
    }
    env = EvaluationPathSelectionEnv(
        topology_data={},
        path_store=store,
        link_states=link_states,
        pair_pool=[(1, 3)],
        episode_length=3,
    )
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    h0 = env.hour_idx
    _, _, done, _ = env.step(0)
    assert env.hour_idx != h0
    assert done is False
    _, _, done, _ = env.step(0)
    assert done is False
    _, _, done, _ = env.step(0)
    assert done is True


def test_action_mask_matches_path_count():
    store = _path_store({(1, 2): [10.0, 20.0], (3, 4): [5.0, 15.0, 25.0]})
    env = EvaluationPathSelectionEnv(
        topology_data={},
        path_store=store,
        link_states={0: {"by_pair": {}}},
        pair_pool=[(1, 2), (3, 4)],
    )
    env.reset(source_as=1, dest_as=2, hour_idx=0)
    mask = env.action_mask(action_dim=4)
    assert mask.tolist() == [True, True, False, False]
    env.reset(source_as=3, dest_as=4, hour_idx=0)
    mask = env.action_mask(action_dim=4)
    assert mask.tolist() == [True, True, True, False]
