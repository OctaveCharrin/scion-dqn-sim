"""
Unified path-selection environment for the evaluation pipeline (steps 04–05).

Provides selective probing, per-pair link states, built-in reward computation,
and consistent observations for flat DQN (5-D aggregate) and path-scoring DQN
(per-path feature matrix).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Flat DQN: time + aggregate link context only.
FLAT_GLOBAL_DIM = 5
GLOBAL_DIM = FLAT_GLOBAL_DIM  # backward-compatible alias

# Path-scoring DQN: flat context + normalized (src, dst) pair embedding.
PAIR_EMBED_DIM = 2
SCORING_GLOBAL_DIM = FLAT_GLOBAL_DIM + PAIR_EMBED_DIM

# Weight-conditioned path-scoring DQN: scoring global + normalized reward weights.
REWARD_WEIGHT_DIM = 5  # w1, w2, w3, w4, w_probe
CONDITIONAL_SCORING_GLOBAL_DIM = SCORING_GLOBAL_DIM + REWARD_WEIGHT_DIM

# Per path: latency, loss, hops, relative bandwidth, utilization, static bw, trust.
PATH_FEATURE_DIM = 7

PROBE_PENALTY_REF_MS = 500.0

ObservationMode = Literal["flat", "scoring"]


@dataclass
class RewardWeights:
    """Composite goodput + trust reward weights (see ``compute_reward``)."""

    w1: float = 0.7
    w2: float = 0.3
    w3: float = 0.5
    w4: float = 0.5
    w_probe: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]] = None) -> "RewardWeights":
        base = cls()
        if not data:
            return base
        for key in ("w1", "w2", "w3", "w4", "w_probe"):
            if key in data:
                setattr(base, key, float(data[key]))
        return base


DEFAULT_REWARD_WEIGHTS = RewardWeights().to_dict()


def encode_reward_weights(
    weights: Optional[Union[RewardWeights, Mapping[str, float]]] = None,
) -> np.ndarray:
    """Fixed-size encoding of reward weights for conditional policies (dim 5)."""
    w = weights if isinstance(weights, RewardWeights) else RewardWeights.from_mapping(weights)
    return np.array(
        [w.w1, w.w2, w.w3, w.w4, w.w_probe],
        dtype=np.float32,
    )


def reward_from_path_metrics(
    path_metrics: Mapping[str, Any],
    env: "EvaluationPathSelectionEnv",
    *,
    weights: Optional[Mapping[str, float]] = None,
    max_possible_bw: Optional[float] = None,
    probe_cost_ms: float = 0.0,
) -> float:
    """Backward-compatible wrapper around ``EvaluationPathSelectionEnv.compute_reward``."""
    return env.compute_reward(
        path_metrics,
        max_possible_bw=max_possible_bw,
        probe_cost_ms=probe_cost_ms,
        num_probes_in_step=1,
        weights=weights,
    )


def _wrap_path(path_dict: Dict[str, Any]) -> SimpleNamespace:
    hops = path_dict.get("hops") or []
    seq = tuple(int(h["as"]) for h in hops if isinstance(h, dict) and "as" in h)
    ns = SimpleNamespace()
    for k, v in path_dict.items():
        setattr(ns, k, v)
    ns.as_sequence = seq
    return ns


def _path_to_link_keys(path: Any) -> List[Tuple[int, int]]:
    hops = getattr(path, "hops", None)
    if hops is None and isinstance(path, dict):
        hops = path.get("hops") or []
    hops = hops or []
    out: List[Tuple[int, int]] = []
    for i in range(len(hops) - 1):
        a, b = int(hops[i]["as"]), int(hops[i + 1]["as"])
        out.append((a, b) if a <= b else (b, a))
    return out


class EvaluationPathSelectionEnv:
    """Path-selection environment with probing, observations, and reward in one place."""

    def __init__(
        self,
        topology_data: Dict[str, Any],
        path_store: Any,
        link_states: Dict[int, Dict[str, Any]],
        latency_probe_cost_ms: float = 10.0,
        bandwidth_probe_cost_ms: float = 100.0,
        per_hop_probe_cost_ms: float = 0.5,
        per_hop_full_probe_cost_ms: float = 20.0,
        probe_type: str = "adaptive",
        pair_pool: Optional[Sequence[Tuple[int, int]]] = None,
        episode_length: int = 24,
        rng_seed: Optional[int] = None,
        reward_weights: Optional[Union[RewardWeights, Mapping[str, float]]] = None,
        normalize_probe_penalty: bool = True,
        max_as: Optional[int] = None,
    ) -> None:
        self.topology_data = topology_data
        self.path_store = path_store
        self.link_states = link_states
        self.latency_probe_cost_ms = float(latency_probe_cost_ms)
        self.bandwidth_probe_cost_ms = float(bandwidth_probe_cost_ms)
        self.per_hop_probe_cost_ms = float(per_hop_probe_cost_ms)
        self.per_hop_full_probe_cost_ms = float(per_hop_full_probe_cost_ms)
        self.probe_type = probe_type
        self.pair_pool: List[Tuple[int, int]] = (
            [(int(s), int(d)) for (s, d) in pair_pool] if pair_pool else []
        )
        self.episode_length = max(1, int(episode_length))
        self._rng = random.Random(rng_seed)

        if isinstance(reward_weights, RewardWeights):
            self.reward_weights = reward_weights
        else:
            self.reward_weights = RewardWeights.from_mapping(reward_weights)

        self.normalize_probe_penalty = bool(normalize_probe_penalty)
        self._max_as = int(max_as) if max_as is not None else None

        self.current_link_states: Dict[str, Any] = {}
        self.available_paths: List[Any] = []
        self.probed_path_metrics: Dict[int, Dict[str, Any]] = {}
        self.current_flow: Dict[str, Any] = {}
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.last_probe_cost_ms = 0.0
        self.total_probe_cost_ms = 0.0
        self.current_step = 0
        self.hour_idx = 0
        self.episode_start_hour = 0

    # ------------------------------------------------------------------ paths
    def _static_metrics(self, path_idx: int) -> Dict[str, Any]:
        p = self.available_paths[path_idx]
        if isinstance(p, SimpleNamespace):
            sm = getattr(p, "static_metrics", None) or {}
        elif isinstance(p, dict):
            sm = p.get("static_metrics", {})
        else:
            sm = {}
        return dict(sm)

    def _path_metrics_at(self, path_index: int) -> Dict[str, Any]:
        if path_index >= len(self.available_paths):
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": 0.0,
                "loss_rate": 1.0,
                "hop_count": 0,
                "utilization": 1.0,
            }
        st = self.current_link_states.get(f"path_{path_index}", {}) or {}
        sm = self._static_metrics(path_index)
        return {
            "latency_ms": float(st.get("latency_ms", sm.get("total_latency", 50.0))),
            "bandwidth_mbps": float(
                st.get("available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0))
            ),
            "loss_rate": float(st.get("loss_rate", 0.0)),
            "hop_count": int(sm.get("hop_count", 1)),
            "utilization": float(st.get("utilization", 0.0)),
        }

    def _max_path_bandwidth_at_current_hour(self) -> float:
        max_bw = 0.0
        for path_index in range(len(self.available_paths)):
            bw = float(self._path_metrics_at(path_index).get("bandwidth_mbps") or 0.0)
            max_bw = max(max_bw, bw)
        return max_bw

    def _max_as_number(self) -> int:
        if self._max_as is not None and self._max_as > 0:
            return self._max_as
        candidates: List[int] = []
        nodes = self.topology_data.get("nodes") or self.topology_data.get("ases") or []
        for node in nodes:
            if isinstance(node, dict):
                for key in ("as_id", "id", "as"):
                    if key in node:
                        candidates.append(int(node[key]))
                        break
            elif isinstance(node, (int, float)):
                candidates.append(int(node))
        for pair in self.pair_pool:
            candidates.extend((int(pair[0]), int(pair[1])))
        src = self.current_flow.get("src")
        dst = self.current_flow.get("dst")
        if src is not None:
            candidates.append(int(src))
        if dst is not None:
            candidates.append(int(dst))
        return max(candidates) if candidates else 50

    def _effective_probe_cost_ms(
        self, step_probe_cost_ms: float, num_probes_in_step: int
    ) -> float:
        """Per-probe average when normalizing so full-probe baselines are not over-penalized."""
        if self.normalize_probe_penalty:
            return float(step_probe_cost_ms) / max(1, int(num_probes_in_step))
        return float(step_probe_cost_ms)

    # ---------------------------------------------------------------- reward
    def compute_reward(
        self,
        path_metrics: Mapping[str, Any],
        *,
        max_possible_bw: Optional[float] = None,
        probe_cost_ms: float = 0.0,
        num_probes_in_step: int = 1,
        weights: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Composite goodput + trust reward in ``[-1, 1]``."""
        w = RewardWeights.from_mapping(weights) if weights else self.reward_weights
        bw = float(path_metrics.get("bandwidth_mbps") or 0.0)

        if max_possible_bw is None:
            max_possible_bw = self._max_path_bandwidth_at_current_hour()

        if max_possible_bw <= 0.001:
            goodput = 1.0
        else:
            goodput = max(0.0, min(bw / max_possible_bw, 1.0))

        loss = float(path_metrics.get("loss_rate", 0.0))
        delay = min(100.0, float(path_metrics.get("latency_ms", 50.0))) / 100.0
        trust = max(0.0, min(1.0, 1.0 - (w.w3 * loss + w.w4 * delay)))
        base_reward = float(2.0 * (w.w1 * goodput + w.w2 * trust) - 1.0)
        cost = self._effective_probe_cost_ms(probe_cost_ms, num_probes_in_step)
        probe_penalty = min(1.0, cost / PROBE_PENALTY_REF_MS)
        reward = base_reward - (w.w_probe * probe_penalty)
        return max(-1.0, min(1.0, float(reward)))

    # ----------------------------------------------------------- observations
    def observe_flat(self, hour_idx: Optional[int] = None) -> np.ndarray:
        """5-D aggregate context: time, mean utilization, mean trust, congestion fraction."""
        hour_idx = self.hour_idx if hour_idx is None else int(hour_idx)
        day = (hour_idx // 24) % 7
        hour = hour_idx % 24
        f0 = day / 6.0
        f1 = hour / 23.0
        w = self.reward_weights
        states = list(self.current_link_states.values())
        if not states:
            return np.array([f0, f1, 0.0, 0.0, 0.0], dtype=np.float32)
        utils = [float(s.get("utilization", 0.0)) for s in states]
        losses = [float(s.get("loss_rate", 0.0)) for s in states]
        lats = [min(100.0, float(s.get("latency_ms", 50.0))) / 100.0 for s in states]
        trusts = [
            max(0.0, min(1.0, 1.0 - (w.w3 * loss + w.w4 * lat)))
            for loss, lat in zip(losses, lats)
        ]
        f2 = float(np.mean(utils)) if utils else 0.0
        f3 = float(np.mean(trusts)) if trusts else 0.0
        f4 = float(np.mean([1.0 if u > 0.7 else 0.0 for u in utils])) if utils else 0.0
        return np.array([f0, f1, f2, f3, f4], dtype=np.float32)

    def observe_scoring_global(self, hour_idx: Optional[int] = None) -> np.ndarray:
        """Scoring global vector: aggregate context + normalized source/destination AS."""
        base = self.observe_flat(hour_idx)
        max_as = float(self._max_as_number())
        src = float(self.current_flow.get("src", 0))
        dst = float(self.current_flow.get("dst", 0))
        pair_embed = np.array([src / max_as, dst / max_as], dtype=np.float32)
        return np.concatenate([base, pair_embed]).astype(np.float32)

    def observe_scoring(self) -> Dict[str, np.ndarray]:
        """Dict state for path-scoring DQNs: ``global`` (SCORING_GLOBAL_DIM,) and ``paths`` (N, 7)."""
        n = len(self.available_paths)
        if n == 0:
            return {
                "global": self.observe_scoring_global(),
                "paths": np.zeros((0, PATH_FEATURE_DIM), dtype=np.float32),
            }

        w = self.reward_weights
        raw_bws: List[float] = []
        for path_idx in range(n):
            st = self.current_link_states.get(f"path_{path_idx}", {}) or {}
            sm = self._static_metrics(path_idx)
            raw_bws.append(
                float(st.get("available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0)))
            )
        max_bw = max(raw_bws) if raw_bws else 1.0
        if max_bw <= 0.001:
            max_bw = 1.0

        rows = np.zeros((n, PATH_FEATURE_DIM), dtype=np.float32)
        for path_idx in range(n):
            sm = self._static_metrics(path_idx)
            st = self.current_link_states.get(f"path_{path_idx}", {}) or {}
            lat_ms = float(st.get("latency_ms", sm.get("total_latency", 50.0)))
            lat = lat_ms / 100.0
            loss = float(st.get("loss_rate", 0.0))
            hop = float(sm.get("hop_count", 1)) / 20.0
            bw_ratio = raw_bws[path_idx] / max_bw
            util = float(st.get("utilization", 0.0))
            static_bw = float(sm.get("min_bandwidth", 0.0)) / 10000.0
            trust = max(0.0, min(1.0, 1.0 - (w.w3 * loss + w.w4 * min(100.0, lat_ms) / 100.0)))
            rows[path_idx] = (lat, loss, hop, bw_ratio, util, static_bw, trust)
        return {"global": self.observe_scoring_global(), "paths": rows}

    def observe_scoring_conditional(self) -> Dict[str, np.ndarray]:
        """Scoring observation with ``encode_reward_weights`` appended to ``global``."""
        obs = self.observe_scoring()
        wvec = encode_reward_weights(self.reward_weights)
        obs["global"] = np.concatenate([obs["global"], wvec]).astype(np.float32)
        return obs

    def observe(self, mode: ObservationMode = "flat") -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if mode == "scoring":
            return self.observe_scoring()
        return self.observe_flat()

    # ---------------------------------------------------------------- episode
    def reset(
        self,
        source_as: Optional[int] = None,
        dest_as: Optional[int] = None,
        *,
        hour_idx: Optional[int] = None,
    ) -> np.ndarray:
        if source_as is None or dest_as is None:
            if not self.pair_pool:
                raise ValueError(
                    "EvaluationPathSelectionEnv.reset: no source/dest provided "
                    "and pair_pool is empty."
                )
            source_as, dest_as = self._rng.choice(self.pair_pool)

        raw_paths = self.path_store.find_paths(int(source_as), int(dest_as))
        self.available_paths = [
            _wrap_path(p) if isinstance(p, dict) else p for p in raw_paths
        ]
        self.probed_path_metrics.clear()
        self.current_flow = {"src": int(source_as), "dst": int(dest_as)}
        self.current_step = 0
        self.num_latency_probes = 0
        self.num_bandwidth_probes = 0
        self.last_probe_cost_ms = 0.0
        self.total_probe_cost_ms = 0.0

        if hour_idx is None:
            if self.link_states:
                hour_idx = self._rng.choice(sorted(self.link_states.keys()))
            else:
                hour_idx = 0
        self.hour_idx = int(hour_idx)
        self.episode_start_hour = int(hour_idx)
        self._refresh_link_states()
        return self.observe_flat()

    def _refresh_link_states(self) -> None:
        hour = self.link_states.get(self.hour_idx, {}) or {}
        src = self.current_flow.get("src")
        dst = self.current_flow.get("dst")
        per_pair = (hour.get("by_pair") or {}) if isinstance(hour, dict) else {}
        key = f"pair_{int(src)}_{int(dst)}" if src is not None and dst is not None else None
        block = per_pair.get(key) if key else None

        if block:
            self.current_link_states = dict(block)
        else:
            cleaned = {
                k: v for k, v in hour.items() if isinstance(k, str) and k.startswith("path_")
            }
            self.current_link_states = cleaned

    # ------------------------------------------------------------------ probe
    def probe_path_latency(self, path_index: int) -> Dict[str, Any]:
        if path_index >= len(self.available_paths):
            self.last_probe_cost_ms = self.latency_probe_cost_ms
            self.total_probe_cost_ms += self.last_probe_cost_ms
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": None,
                "loss_rate": 1.0,
                "hop_count": 0,
                "probe_type": "latency",
                "probe_cost_ms": self.last_probe_cost_ms,
            }

        sm = self._static_metrics(path_index)
        st = self.current_link_states.get(f"path_{path_index}", {}) or {}
        lat = float(st.get("latency_ms", sm.get("total_latency", 50.0)))
        loss = float(st.get("loss_rate", 0.0))
        hop = int(sm.get("hop_count", 1))
        cost = self.latency_probe_cost_ms + self.per_hop_probe_cost_ms * hop

        self.num_latency_probes += 1
        self.last_probe_cost_ms = cost
        self.total_probe_cost_ms += cost

        return {
            "latency_ms": lat,
            "bandwidth_mbps": None,
            "loss_rate": loss,
            "hop_count": hop,
            "probe_type": "latency",
            "probe_cost_ms": cost,
        }

    def probe_path_full(self, path_index: int) -> Dict[str, Any]:
        if path_index >= len(self.available_paths):
            self.last_probe_cost_ms = self.bandwidth_probe_cost_ms
            self.total_probe_cost_ms += self.last_probe_cost_ms
            return {
                "latency_ms": float("inf"),
                "bandwidth_mbps": 0.0,
                "loss_rate": 1.0,
                "hop_count": 0,
                "probe_type": "full",
                "probe_cost_ms": self.last_probe_cost_ms,
            }

        sm = self._static_metrics(path_index)
        st = self.current_link_states.get(f"path_{path_index}", {}) or {}
        lat = float(st.get("latency_ms", sm.get("total_latency", 50.0)))
        bw = float(
            st.get("available_bandwidth_mbps", sm.get("min_bandwidth", 1000.0))
        )
        loss = float(st.get("loss_rate", 0.0))
        hop = int(sm.get("hop_count", 1))
        cost = self.bandwidth_probe_cost_ms + self.per_hop_full_probe_cost_ms * hop

        self.num_latency_probes += 1
        self.num_bandwidth_probes += 1
        self.last_probe_cost_ms = cost
        self.total_probe_cost_ms += cost

        out = {
            "latency_ms": lat,
            "bandwidth_mbps": bw,
            "loss_rate": loss,
            "hop_count": hop,
            "probe_type": "full",
            "probe_cost_ms": cost,
        }
        self.probed_path_metrics[path_index] = out
        return out

    # ------------------------------------------------------------------ step
    def step(
        self,
        action: int,
        *,
        step_probe_cost_ms: float = 0.0,
        num_probes_in_step: int = 1,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply ``action`` at the current hour, compute reward, then advance time.

        ``max_available_path_bw`` and ``path_metrics`` are taken at the selection
        hour (before ``hour_idx`` advances). Pass ``num_probes_in_step`` so probe
        penalty is comparable between single-probe RL and multi-probe baselines.
        """
        self.current_step += 1
        selection_hour = self.hour_idx
        path_metrics = self._path_metrics_at(action)
        max_available_path_bw = self._max_path_bandwidth_at_current_hour()
        effective_probe_cost = self._effective_probe_cost_ms(
            step_probe_cost_ms, num_probes_in_step
        )
        reward = self.compute_reward(
            path_metrics,
            max_possible_bw=max_available_path_bw,
            probe_cost_ms=step_probe_cost_ms,
            num_probes_in_step=num_probes_in_step,
        )

        keys = sorted(self.link_states.keys()) if self.link_states else []
        if keys:
            self.hour_idx = (self.hour_idx + 1) % (max(keys) + 1)
        self._refresh_link_states()

        done = self.current_step >= self.episode_length
        info = {
            "path_metrics": path_metrics,
            "max_available_path_bw": max_available_path_bw,
            "reward": reward,
            "selection_hour_idx": selection_hour,
            "probe_count": self.num_latency_probes + self.num_bandwidth_probes,
            "probe_cost_ms": self.total_probe_cost_ms,
            "step_probe_cost_ms": float(step_probe_cost_ms),
            "effective_probe_cost_ms": float(effective_probe_cost),
            "num_probes_in_step": int(num_probes_in_step),
            "hour_idx": self.hour_idx,
            "action": int(action),
        }
        return self.observe_flat(), reward, done, info

    def apply_action(
        self,
        action: int,
        *,
        probe: Literal["none", "latency", "full"] = "full",
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Optionally probe the chosen path, then ``step`` with matching probe cost."""
        step_probe_cost = 0.0
        num_probes = 0
        if probe == "full":
            self.probe_path_full(action)
            step_probe_cost = self.last_probe_cost_ms
            num_probes = 1
        elif probe == "latency":
            self.probe_path_latency(action)
            step_probe_cost = self.last_probe_cost_ms
            num_probes = 1
        _, reward, done, info = self.step(
            action,
            step_probe_cost_ms=step_probe_cost,
            num_probes_in_step=max(1, num_probes),
        )
        return reward, done, info

    def num_paths(self) -> int:
        return len(self.available_paths)

    def action_mask(self, action_dim: int) -> np.ndarray:
        n = len(self.available_paths)
        mask = np.zeros(int(action_dim), dtype=bool)
        mask[: min(n, int(action_dim))] = True
        return mask
