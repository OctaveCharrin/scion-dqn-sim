"""Tests for the evaluation-pipeline simulation helpers."""

from __future__ import annotations

from src.simulation.path_store import InMemoryPathStore


def test_path_store_roundtrip():
    store = InMemoryPathStore()
    assert store.find_paths(1, 2) == []

    paths = [{"hops": [1, 2]}, {"hops": [1, 3, 2]}]
    store.set_paths(1, 2, paths)
    got = store.find_paths(1, 2)
    assert got == paths
    # Mutating the returned list must not affect the store's internal state.
    got.append({"hops": [1, 4, 2]})
    assert len(store.find_paths(1, 2)) == 2


def test_path_store_coerces_int_keys():
    store = InMemoryPathStore()
    store.set_paths(7, 9, [{"hops": [7, 8, 9]}])
    # String keys that repr as ints still resolve through int() coercion.
    assert len(store.find_paths("7", "9")) == 1


def test_path_store_missing_pair_is_empty_list():
    store = InMemoryPathStore()
    store.set_paths(1, 2, [{"hops": [1, 2]}])
    assert store.find_paths(2, 1) == []
