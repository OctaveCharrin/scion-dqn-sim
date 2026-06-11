"""Unit tests for ``Ns3DataPlane`` that do **not** require a built NS-3 tree.

These exercise the pure-Python boundary logic (e.g. seed handling) with the
shared-memory bridge stubbed out, so they run anywhere — unlike
``tests/test_ns3_dataplane_integration.py`` which drives a real NS-3 process.
"""

from __future__ import annotations

from src.ns3env import Ns3Config, Ns3DataPlane


def _stub_bridge(dp: Ns3DataPlane) -> None:
    """Neutralize the parts of ``reset`` that would touch a real NS-3 process."""
    dp._launch = lambda: setattr(dp, "_started", True)  # type: ignore[assignment]
    dp._recv_env = lambda: None  # type: ignore[assignment]
    dp._send_act = lambda *a, **k: None  # type: ignore[assignment]


def test_reset_clamps_seed_zero_to_one():
    # NS-3's RngSeedManager fatally rejects seed 0; callers routinely pass
    # seed=base+episode with base 0, so reset() must clamp it into [1, ...).
    dp = Ns3DataPlane(config=Ns3Config(seed=5))
    _stub_bridge(dp)
    dp.reset(seed=0)
    assert dp.config.seed == 1


def test_reset_preserves_valid_seed():
    dp = Ns3DataPlane(config=Ns3Config(seed=5))
    _stub_bridge(dp)
    dp.reset(seed=7)
    assert dp.config.seed == 7
