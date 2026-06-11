"""Integration tests for the real NS-3 ``Ns3DataPlane`` backend.

These only run where the ``video_mpquic`` scenario has been built (Linux/WSL2 +
ns3-ai; see ``ns3/README.md``). Everywhere else they skip, so CI on the pure
mock backend is unaffected. Point $NS3_DIR at the built ns-3 tree (defaults to
``~/ns-3-dev``).

ns3-ai permits only **one** shared-memory creator (one ``Experiment``) per
Python process, so every test shares a single module-scoped ``Ns3DataPlane`` and
exercises episodes via in-band ``reset()`` on the same long-lived NS-3 process.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from src.ns3env import (
    GLOBAL_DIM,
    PATH_FEATURE_DIM,
    Ns3Config,
    Ns3DataPlane,
    Ns3VideoMpquicEnv,
)

NS3_DIR = os.path.abspath(os.environ.get("NS3_DIR", os.path.expanduser("~/ns-3-dev")))
_EXAMPLE_DIR = os.path.join(NS3_DIR, "contrib", "ai", "examples", "video-mpquic")
_SEGMENTS = 4
_SEGMENT_BYTES = 200_000


def _ns3_built() -> bool:
    """True only if both the pybind module and the scenario binary exist."""
    has_pyso = bool(glob.glob(os.path.join(_EXAMPLE_DIR, "ns3ai_video_mpquic_py*.so")))
    has_bin = bool(
        glob.glob(
            os.path.join(
                NS3_DIR,
                "build",
                "contrib",
                "ai",
                "examples",
                "video-mpquic",
                "*ns3ai_video_mpquic*",
            )
        )
    )
    try:
        import ns3ai_utils  # noqa: F401
    except ImportError:
        return False
    return has_pyso and has_bin


pytestmark = pytest.mark.skipif(
    not _ns3_built(),
    reason="NS-3 video_mpquic scenario not built (see ns3/README.md)",
)


@pytest.fixture(scope="module")
def dp():
    """One NS-3 process shared by all tests (one ns3-ai creator per process)."""
    plane = Ns3DataPlane(
        config=Ns3Config(
            episode_segments=_SEGMENTS, segment_bytes=_SEGMENT_BYTES, seed=1
        )
    )
    yield plane
    plane.close()


def _run_episode(plane: Ns3DataPlane):
    """Drive a full episode with a round-robin policy; return the results."""
    results = []
    step = 0
    while not plane.is_done() and step < plane.config.episode_segments + 5:
        action = step % plane.num_paths
        results.append(plane.download_segment(action, _SEGMENT_BYTES))
        step += 1
    return results


def test_reset_returns_initial_observation(dp):
    dp.reset(seed=1)
    assert dp.num_paths == 3
    stats = dp.current_path_stats()
    assert len(stats) == 3
    # Initial estimates seed to nominal link rate / base RTT.
    assert all(s.throughput_mbps > 0 for s in stats)
    assert all(s.rtt_ms > 0 for s in stats)
    assert dp.clock_s >= 0.0
    assert dp.is_done() is False


def test_full_episode_delivers_every_segment(dp):
    dp.reset()
    results = _run_episode(dp)
    assert len(results) == _SEGMENTS
    assert dp.is_done() is True
    for r in results:
        assert r.bytes_delivered >= _SEGMENT_BYTES
        assert r.throughput_mbps > 0.0
        assert r.rtt_ms > 0.0
        assert 0.0 <= r.loss <= 1.0
        assert r.duration_s > 0.0


def test_multiple_in_band_resets(dp):
    """Training resets every episode; each reset reuses the same NS-3 process."""
    dp.reset()
    r1 = _run_episode(dp)
    dp.reset()
    r2 = _run_episode(dp)
    dp.reset()
    r3 = _run_episode(dp)
    assert len(r1) == _SEGMENTS
    assert len(r2) == _SEGMENTS
    assert len(r3) == _SEGMENTS
    # The simulation is continuing, so the clock advances across episodes.
    assert dp.clock_s > 0.0


def test_env_observation_contract_matches_mock(dp):
    """The NS-3 backend yields the same {global,paths} shapes the agents expect."""
    env = Ns3VideoMpquicEnv(dp)
    obs = env.reset(seed=1)
    assert obs["global"].shape == (GLOBAL_DIM,)
    assert obs["paths"].shape == (dp.num_paths, PATH_FEATURE_DIM)
    assert obs["paths"].dtype == np.float32

    obs2, reward, done, info = env.step(0)
    assert obs2["paths"].shape == (dp.num_paths, PATH_FEATURE_DIM)
    assert isinstance(reward, float)
    assert info["chosen_path"] == 0
    assert info["bytes_delivered"] >= _SEGMENT_BYTES
