"""
Train + compare path-selection policies for the multipath-QUIC video env.

This is the Phase-1 validation harness: it trains the scoring DQN, then evaluates
it alongside every heuristic baseline in
:data:`~src.baselines.multipath_baselines.MULTIPATH_SELECTORS` on the *same*
scenario, and reports mean per-step reward and goodput per method. It is the seed
of the Phase-3 numbered evaluation pipeline (``ns3/PLAN.md``); for now it is a
single self-contained driver.

Backends
--------
One :class:`~src.ns3env.dataplane.DataPlane` is created and **shared** across
training and all evaluations (the factory returns the same instance). This is
required for the NS-3 backend (ns3-ai permits only one shared-memory ``Experiment``
per process) and is harmless for the mock backend.

Fairness note
-------------
On ``MockTraceDataPlane`` every method sees an identical scenario: ``reset(seed)``
rewinds the clock and reseeds deterministically, and each method is evaluated with
the same seed base, so the comparison is apples-to-apples. On ``Ns3DataPlane`` the
simulation is *continuing* (in-band resets keep the network evolving), so the
methods are evaluated on consecutive — not identical — slices of one run; treat
NS-3 numbers as indicative until a fair-replay protocol is added.

Usage
-----
    uv run python -m src.rl.video_eval_compare --episodes 300 --eval-episodes 30
    # NS-3 backend (built tree required; see ns3/README.md):
    NS3_DIR=~/ns-3-dev uv run python -m src.rl.video_eval_compare --backend ns3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.baselines.multipath_baselines import MULTIPATH_SELECTORS
from src.ns3env.dataplane import (
    DataPlane,
    MockTraceConfig,
    MockTraceDataPlane,
)
from src.ns3env.reward import RewardWeights
from src.rl.video_mpquic_train import (
    DataPlaneFactory,
    evaluate_policy,
    train_video_dqn,
)

# Mock scenario presets. ``default`` is the asymmetric one-dominant-path setting
# (where "always pick the strongest path" is already near-optimal); ``crossover``
# gives three equal-mean, high-amplitude paths whose phases are auto-spread, so the
# best path rotates over time and any static / single-path policy is suboptimal --
# the setting that actually exercises a state-aware agent.
_SCENARIOS: Dict[str, MockTraceConfig] = {
    "default": MockTraceConfig(),
    "crossover": MockTraceConfig(
        num_paths=3,
        base_mbps=(4.0, 4.0, 4.0),
        amp=(0.85, 0.85, 0.85),
        period_s=(24.0, 24.0, 24.0),
        base_rtt_ms=(25.0, 25.0, 25.0),
        noise_std=0.05,
    ),
}


def _make_dataplane(backend: str, scenario: str = "default") -> DataPlane:
    if backend == "mock":
        cfg = _SCENARIOS.get(scenario)
        if cfg is None:
            raise ValueError(f"unknown scenario {scenario!r} ({list(_SCENARIOS)})")
        return MockTraceDataPlane(cfg)
    if backend == "ns3":
        # Imported lazily so the mock path has no NS-3 dependency.
        from src.ns3env.dataplane import Ns3Config, Ns3DataPlane

        return Ns3DataPlane(config=Ns3Config())
    raise ValueError(f"unknown backend {backend!r} (use 'mock' or 'ns3')")


def run_comparison(
    *,
    backend: str = "mock",
    scenario: str = "default",
    num_episodes: int = 300,
    eval_episodes: int = 30,
    seed: int = 42,
    eval_seed: int = 10_000,
    reward_weights: Optional[RewardWeights] = None,
    out_path: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Train the DQN and evaluate it against all baselines on one shared data plane.

    Returns a results dict (also written to ``out_path`` as JSON when given).
    """
    reward_weights = reward_weights or RewardWeights()
    dataplane = _make_dataplane(backend, scenario)
    # The SAME data plane backs training and every evaluation (see module docstring).
    factory: DataPlaneFactory = lambda: dataplane  # noqa: E731

    train_stats, agent = train_video_dqn(
        dataplane_factory=factory,
        num_episodes=num_episodes,
        reward_weights=reward_weights,
        seed=seed,
        quiet=quiet,
    )

    methods: Dict[str, Dict[str, float]] = {}
    methods["dqn"] = evaluate_policy(
        agent,
        dataplane_factory=factory,
        reward_weights=reward_weights,
        episodes=eval_episodes,
        seed=eval_seed,
        is_agent=True,
    )
    for name, selector_cls in MULTIPATH_SELECTORS.items():
        methods[name] = evaluate_policy(
            selector_cls(),
            dataplane_factory=factory,
            reward_weights=reward_weights,
            episodes=eval_episodes,
            seed=eval_seed,
            is_agent=False,
        )

    # DQN headroom over the strongest baseline.
    best_baseline = max(
        (n for n in methods if n != "dqn"),
        key=lambda n: methods[n]["mean_reward"],
    )
    dqn_r = methods["dqn"]["mean_reward"]
    base_r = methods[best_baseline]["mean_reward"]

    results: Dict[str, Any] = {
        "backend": backend,
        "scenario": scenario,
        "num_train_episodes": num_episodes,
        "eval_episodes": eval_episodes,
        "cap_mbps": train_stats.get("cap_mbps"),
        "train_avg_reward_last_decile": train_stats.get("avg_reward_last_decile"),
        "methods": methods,
        "best_baseline": best_baseline,
        "dqn_minus_best_baseline_reward": dqn_r - base_r,
        "dqn_beats_best_baseline": dqn_r > base_r,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        results["out_path"] = str(out_path)

    if backend == "ns3" and hasattr(dataplane, "close"):
        dataplane.close()

    return results


def _print_table(results: Dict[str, Any]) -> None:
    methods = results["methods"]
    order = sorted(methods, key=lambda n: methods[n]["mean_reward"], reverse=True)
    width = max(len(n) for n in methods)
    print(
        f"\nBackend: {results['backend']}  scenario: {results.get('scenario')}  "
        f"(train {results['num_train_episodes']} ep, "
        f"eval {results['eval_episodes']} ep)"
    )
    print(f"{'method':<{width}}  {'mean_reward':>12}  {'goodput_mbps':>13}")
    print("-" * (width + 29))
    for name in order:
        m = methods[name]
        star = "  <- DQN" if name == "dqn" else ""
        print(
            f"{name:<{width}}  {m['mean_reward']:>12.4f}  "
            f"{m['mean_goodput_mbps']:>13.4f}{star}"
        )
    verdict = "BEATS" if results["dqn_beats_best_baseline"] else "does NOT beat"
    print(
        f"\nDQN {verdict} best baseline '{results['best_baseline']}' "
        f"by {results['dqn_minus_best_baseline_reward']:+.4f} reward."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["mock", "ns3"], default="mock")
    p.add_argument(
        "--scenario",
        choices=list(_SCENARIOS),
        default="default",
        help="mock scenario preset (ignored for the ns3 backend)",
    )
    p.add_argument("--episodes", type=int, default=300, help="training episodes")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None, help="write results JSON here")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    results = run_comparison(
        backend=args.backend,
        scenario=args.scenario,
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        out_path=args.out,
        quiet=args.quiet,
    )
    _print_table(results)


if __name__ == "__main__":
    main()
