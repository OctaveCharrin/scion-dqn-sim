"""Tunable parameters for link-level traffic simulation (pipeline step 03)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class TrafficSimConfig:
    """Calibrated traffic demand and congestion targets.

    Design goals (see project traffic calibration notes):

    * Sparse foreground: only ``active_pairs_min``–``active_pairs_max`` pairs
      inject demand each hour (not the full ``pair_pool``).
    * Demand scales with topology size via ``reference_pair_pool_size``.
    * Background load scales with topology size and active foreground count.
    * Target band: p90 link utilization ~0.6–0.85 at peak; 10–30% of
      (pair, hour) with zero max path bandwidth under stress (not ~90%).
    """

    num_days: int = 28
    samples_per_day: int = 24

    base_rate_mbps: float = 100.0
    """Nominal mean demand (Mbps) for a reference-sized pair pool."""

    reference_pair_pool_size: int = 32
    """Scale ``base_rate_mbps`` as if the pool had this many pairs."""

    active_pairs_min: int = 50
    active_pairs_max: int = 150

    background_min: int = 20
    background_max: int = 50
    """Cap background flows per hour (also bounded by ``n_nodes``)."""

    background_mbps_min: float = 10.0
    background_mbps_max: float = 80.0

    elephant_fraction: float = 0.12
    """Fraction of active pairs treated as heavy hitters each hour."""

    elephant_rate_multiplier: float = 3.0

    top_paths_per_pair: int = 3
    util_cap_in_path_metrics: float = 1.5

    target_p90_utilization: float = 0.75
    target_zero_bw_pair_hour_fraction: float = 0.20

    prop_rng_seed: int = 42
    write_link_states_json: bool = False

    @property
    def total_hours(self) -> int:
        return self.num_days * self.samples_per_day

    def scaled_base_rate_mbps(self, pair_pool_size: int) -> float:
        """Reduce per-pair rate when ``pair_pool`` is large."""
        pool = max(1, int(pair_pool_size))
        return float(self.base_rate_mbps) * (
            float(self.reference_pair_pool_size) / float(pool)
        )

    def background_pairs_per_hour(
        self, n_nodes: int, active_foreground: int
    ) -> int:
        cap = min(self.background_max, max(self.background_min, n_nodes))
        tied = max(self.background_min, active_foreground // 2)
        return int(min(cap, tied))

    @classmethod
    def from_env(cls) -> "TrafficSimConfig":
        cfg = cls()
        mapping = {
            "TRAFFIC_BASE_RATE_MBPS": ("base_rate_mbps", float),
            "TRAFFIC_ACTIVE_PAIRS_MIN": ("active_pairs_min", int),
            "TRAFFIC_ACTIVE_PAIRS_MAX": ("active_pairs_max", int),
            "TRAFFIC_BG_MAX": ("background_max", int),
            "TRAFFIC_WRITE_JSON": ("write_link_states_json", lambda v: v == "1"),
        }
        for env_key, (attr, cast) in mapping.items():
            val = os.environ.get(env_key, "").strip()
            if val:
                setattr(cfg, attr, cast(val))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
