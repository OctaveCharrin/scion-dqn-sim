"""Tests for the traffic engine and link-annotation utilities."""

from __future__ import annotations

from src.link_annotation.capacity_delay_builder import CapacityDelayBuilder
from src.traffic.traffic_engine import TrafficEngine


def test_diurnal_pattern_has_peaks_and_troughs():
    engine = TrafficEngine()
    values = [engine._diurnal_pattern(h) for h in range(24)]

    assert min(values) >= 0.0
    assert max(values) <= 1.5
    # Meaningful variation between busiest and quietest hours.
    assert max(values) > min(values) * 1.5


def test_diurnal_pattern_varies_by_hour():
    engine = TrafficEngine()
    # Pattern must produce at least two distinct values during a day.
    uniques = {round(engine._diurnal_pattern(h), 6) for h in range(24)}
    assert len(uniques) > 1


def test_queueing_delay_monotonic_and_bounded():
    rtt_min = 10.0
    prev = -1.0
    for util in [0.0, 0.2, 0.5, 0.7, 0.9, 0.95]:
        delay = CapacityDelayBuilder.queueing_delay(util, rtt_min)
        assert delay >= 0.0
        assert delay >= prev  # non-decreasing with utilization
        prev = delay
        assert delay < 100 * rtt_min  # bounded below the asymptote


def test_queueing_delay_zero_at_zero_utilization():
    assert CapacityDelayBuilder.queueing_delay(0.0, 10.0) == 0.0
