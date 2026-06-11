"""Tests for the ABR building blocks: VMAF model, buffer, QoE reward."""

from __future__ import annotations

import math

from src.ns3env.abr import (
    BITRATE_LADDER_MBPS,
    SEGMENT_DURATION_S,
    VMAF_MAX,
    PlaybackBuffer,
    QoEWeights,
    compute_qoe_reward,
    segment_bytes_for,
    vmaf_for,
)


def test_vmaf_monotonic_concave_and_bounded():
    ladder = BITRATE_LADDER_MBPS
    scores = [vmaf_for(b) for b in ladder]
    # In range and strictly increasing with bitrate.
    assert all(0.0 <= s <= VMAF_MAX for s in scores)
    assert all(b < a for b, a in zip(scores, scores[1:]))
    # Concave: equal *ratio* steps give diminishing VMAF gains (log curve), so
    # successive absolute increments shrink.
    incr = [a - b for b, a in zip(scores, scores[1:])]
    # The ladder ratios shrink toward the top, so increments are non-increasing-ish;
    # check the top step gains less than the bottom step.
    assert incr[-1] < incr[0]


def test_vmaf_high_bitrate_saturates_near_excellent():
    assert vmaf_for(4.3) >= 90.0
    assert vmaf_for(0.3) <= 35.0


def test_segment_bytes_scales_with_bitrate_and_duration():
    assert segment_bytes_for(1.0, 8.0) == segment_bytes_for(2.0, 4.0)
    assert segment_bytes_for(4.3, 4.0) == int(round(4.3e6 * 4.0 / 8.0))


def test_buffer_fills_when_download_fast():
    buf = PlaybackBuffer(segment_duration_s=4.0, buffer_max_s=60.0, initial_s=4.0)
    # Download faster than playback against a non-empty buffer: no stall, grows.
    rebuf = buf.update(download_time_s=1.0)
    assert rebuf == 0.0
    assert math.isclose(buf.buffer_s, 7.0)  # 4 - 1 + 4
    rebuf2 = buf.update(download_time_s=1.0)
    assert rebuf2 == 0.0
    assert math.isclose(buf.buffer_s, 10.0)  # 7 - 1 + 4


def test_buffer_startup_stall_from_empty():
    buf = PlaybackBuffer(segment_duration_s=4.0, buffer_max_s=60.0)
    # Empty buffer: the first download stalls for its whole duration (startup).
    rebuf = buf.update(download_time_s=1.0)
    assert math.isclose(rebuf, 1.0)
    assert math.isclose(buf.buffer_s, 4.0)  # drained to 0, then +4


def test_buffer_stalls_when_download_slow():
    buf = PlaybackBuffer(segment_duration_s=4.0, buffer_max_s=60.0, initial_s=2.0)
    # 5s download against 2s buffer -> 3s rebuffering.
    rebuf = buf.update(download_time_s=5.0)
    assert math.isclose(rebuf, 3.0)
    assert math.isclose(buf.buffer_s, 4.0)  # drained to 0, then +4


def test_buffer_capped_at_max():
    buf = PlaybackBuffer(segment_duration_s=4.0, buffer_max_s=5.0, initial_s=4.0)
    buf.update(download_time_s=0.0)  # 4 - 0 + 4 = 8 -> capped to 5
    assert buf.buffer_s == 5.0


def test_qoe_reward_in_range_and_quality_increasing():
    # No rebuffer, no switch: higher bitrate -> higher (VMAF-based) reward.
    r_low = compute_qoe_reward(bitrate_mbps=0.3, prev_bitrate_mbps=0.3, rebuffer_s=0.0)
    r_high = compute_qoe_reward(bitrate_mbps=4.3, prev_bitrate_mbps=4.3, rebuffer_s=0.0)
    assert -1.0 <= r_low <= 1.0 and -1.0 <= r_high <= 1.0
    assert r_high > r_low


def test_qoe_rebuffer_and_switch_penalize():
    base = compute_qoe_reward(bitrate_mbps=2.85, prev_bitrate_mbps=2.85, rebuffer_s=0.0)
    with_rebuf = compute_qoe_reward(
        bitrate_mbps=2.85, prev_bitrate_mbps=2.85, rebuffer_s=SEGMENT_DURATION_S
    )
    with_switch = compute_qoe_reward(
        bitrate_mbps=2.85, prev_bitrate_mbps=0.3, rebuffer_s=0.0
    )
    assert with_rebuf < base
    assert with_switch < base


def test_qoe_weights_zeroing_switch_removes_penalty():
    w = QoEWeights(w_switch=0.0)
    a = compute_qoe_reward(
        bitrate_mbps=2.85, prev_bitrate_mbps=0.3, rebuffer_s=0.0, weights=w
    )
    b = compute_qoe_reward(
        bitrate_mbps=2.85, prev_bitrate_mbps=2.85, rebuffer_s=0.0, weights=w
    )
    assert math.isclose(a, b)
