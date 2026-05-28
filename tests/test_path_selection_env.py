"""Tests for unified observations and reward timing in EvaluationPathSelectionEnv."""

from __future__ import annotations

from src.simulation.evaluation_env import (
    FLAT_GLOBAL_DIM,
    PATH_FEATURE_DIM,
    SCORING_GLOBAL_DIM,
    EvaluationPathSelectionEnv,
    RewardWeights,
)
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
    out = {}
    for hour, blocks in pair_states.items():
        out[hour] = {"by_pair": {f"pair_{p[0]}_{p[1]}": st for p, st in blocks.items()}}
    return out


def test_observe_flat_and_scoring_shapes():
    store = _path_store({(1, 3): [25.0, 40.0]})
    link_states = _link_states_with_pair(
        {
            0: {
                (1, 3): {
                    "path_0": {
                        "latency_ms": 25.0,
                        "loss_rate": 0.0,
                        "utilization": 0.1,
                        "available_bandwidth_mbps": 100.0,
                    },
                    "path_1": {
                        "latency_ms": 40.0,
                        "loss_rate": 0.0,
                        "utilization": 0.2,
                        "available_bandwidth_mbps": 200.0,
                    },
                }
            }
        }
    )
    env = EvaluationPathSelectionEnv({}, store, link_states, pair_pool=[(1, 3)])
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    flat = env.observe_flat()
    scoring = env.observe_scoring()
    assert flat.shape == (FLAT_GLOBAL_DIM,)
    assert scoring["global"].shape == (SCORING_GLOBAL_DIM,)
    assert scoring["paths"].shape == (2, PATH_FEATURE_DIM)
    assert scoring["paths"][1, 3] > scoring["paths"][0, 3]  # relative bw


def test_scoring_global_includes_pair_embedding():
    store = _path_store({(5, 9): [25.0]})
    link_states = _link_states_with_pair(
        {0: {(5, 9): {"path_0": {"latency_ms": 25.0, "available_bandwidth_mbps": 100.0}}}}
    )
    env = EvaluationPathSelectionEnv(
        {}, store, link_states, pair_pool=[(5, 9)], max_as=20
    )
    env.reset(source_as=5, dest_as=9, hour_idx=0)
    g = env.observe_scoring_global()
    assert g[-2] == 5 / 20.0
    assert g[-1] == 9 / 20.0


def test_step_reward_uses_selection_hour_bandwidth_cap():
    store = _path_store({(1, 3): [25.0, 40.0]})
    link_states = {
        0: {
            "by_pair": {
                "pair_1_3": {
                    "path_0": {
                        "latency_ms": 25.0,
                        "loss_rate": 0.0,
                        "utilization": 0.0,
                        "available_bandwidth_mbps": 100.0,
                    },
                    "path_1": {
                        "latency_ms": 40.0,
                        "loss_rate": 0.0,
                        "utilization": 0.0,
                        "available_bandwidth_mbps": 500.0,
                    },
                }
            }
        },
        1: {
            "by_pair": {
                "pair_1_3": {
                    "path_0": {
                        "latency_ms": 25.0,
                        "loss_rate": 0.0,
                        "utilization": 0.0,
                        "available_bandwidth_mbps": 1000.0,
                    },
                    "path_1": {
                        "latency_ms": 40.0,
                        "loss_rate": 0.0,
                        "utilization": 0.0,
                        "available_bandwidth_mbps": 1000.0,
                    },
                }
            }
        },
    }
    env = EvaluationPathSelectionEnv({}, store, link_states, pair_pool=[(1, 3)], episode_length=1)
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    _, reward, _, info = env.step(0, step_probe_cost_ms=0.0)
    assert info["selection_hour_idx"] == 0
    assert info["max_available_path_bw"] == 500.0
    assert info["path_metrics"]["bandwidth_mbps"] == 100.0
    manual = env.compute_reward(info["path_metrics"], max_possible_bw=500.0, probe_cost_ms=0.0)
    assert abs(reward - manual) < 1e-6


def test_normalized_probe_penalty():
    store = _path_store({(1, 3): [25.0]})
    link_states = _link_states_with_pair(
        {0: {(1, 3): {"path_0": {"latency_ms": 25.0, "available_bandwidth_mbps": 250.0}}}}
    )
    env = EvaluationPathSelectionEnv(
        {}, store, link_states, pair_pool=[(1, 3)], normalize_probe_penalty=True
    )
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    pm = env._path_metrics_at(0)
    r_one = env.compute_reward(pm, probe_cost_ms=500.0, num_probes_in_step=1)
    r_ten = env.compute_reward(pm, probe_cost_ms=500.0, num_probes_in_step=10)
    assert r_ten > r_one


def test_apply_action_probe_and_reward():
    store = _path_store({(1, 3): [25.0]})
    link_states = _link_states_with_pair(
        {0: {(1, 3): {"path_0": {"latency_ms": 30.0, "available_bandwidth_mbps": 250.0, "loss_rate": 0.05, "utilization": 0.4}}}}
    )
    env = EvaluationPathSelectionEnv(
        {}, store, link_states, pair_pool=[(1, 3)], episode_length=1
    )
    env.reset(source_as=1, dest_as=3, hour_idx=0)
    reward, done, info = env.apply_action(0, probe="full")
    assert done is True
    assert info["step_probe_cost_ms"] > 0
    assert -1.0 <= reward <= 1.0
