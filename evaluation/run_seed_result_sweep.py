#!/usr/bin/env python3
"""Re-run seed-specific per-result studies once per training seed.

Writes ``<run>/seeds/seed<N>/shipped/{intent,zeroshot,pathcount,probing}/``.
Aggregate the result with ``analyze_seed_results.py``.

Usage: uv run python run_seed_result_sweep.py <run_dir> [--seeds 1 2 3] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import eval_intent_interpolation as interp
import eval_pathcount_scaling as pathcount

from src.pipeline.intent_cond_eval import (
    CONDITIONAL_CHECKPOINTS,
    INTENT_PROFILES,
    MAX_EVAL_PAIRS,
    run_intent_alignment,
    run_probing_ceiling,
)
from src.pipeline.run_dirs import resolve_run_dir
from src.simulation.run_context import load_run_context

# The conditioning variant the thesis ships (sec:p1eval:ablation, rung four).
SHIPPED_AGENT = "conditional_concat_2stream"
# Checkpoints a seed's staging dir exposes: the four rungs of the ladder, and
# deliberately not ``dqn_conditional_scoring_model.pth`` (FiLM, dropped).
SHIPPED_CHECKPOINTS: List[str] = [
    "dqn_model.pth",
    "dqn_scoring_enhanced_model.pth",
    CONDITIONAL_CHECKPOINTS["conditional_concat"],
    CONDITIONAL_CHECKPOINTS[SHIPPED_AGENT],
]
# Artifacts shared by every seed; linked so a staging dir also works standalone.
SHARED_ARTIFACTS: List[str] = [
    "topology",
    "selected_pair.json",
    "path_store.json",
    "link_states.pkl",
]

STUDIES = ("intent", "zeroshot", "pathcount", "probing")


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def stage_seed(run_path: Path, seed_dir: Path) -> Path:
    """A run-dir-shaped view of one seed holding only the reported checkpoints."""
    staged = seed_dir / "shipped"
    staged.mkdir(parents=True, exist_ok=True)
    for name in SHARED_ARTIFACTS:
        source = seed_dir / name
        _link(source if source.exists() else run_path / name, staged / name)
    missing = []
    for ckpt in SHIPPED_CHECKPOINTS:
        source = seed_dir / ckpt
        if source.exists():
            _link(source, staged / ckpt)
        else:
            missing.append(ckpt)
    if missing:
        raise FileNotFoundError(
            f"{seed_dir} is missing {', '.join(missing)}. Run run_seed_sweep.sh "
            f"first so every seed carries all four rungs of the ladder."
        )
    return staged


def _done(out_dir: Path, *files: str) -> bool:
    return all((out_dir / f).is_file() for f in files)


def run_seed(
    staged: Path,
    ctx: tuple,
    *,
    max_pairs: int,
    studies: Sequence[str],
    force: bool,
    congestion_bins: int,
    permutations: int,
    progress=print,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "intent" in studies:
        d = staged / "intent"
        if force or not _done(
            d, "intent_reward_matrix.csv", "intent_selection_metrics.csv"
        ):
            t0 = time.time()
            run_intent_alignment(
                staged,
                d,
                max_pairs=max_pairs,
                progress=progress,
                run_context=ctx,
                agent_key=SHIPPED_AGENT,
            )
            out["intent_s"] = round(time.time() - t0, 1)
        else:
            progress("  intent alignment: cached")

    if "zeroshot" in studies:
        d = staged / "zeroshot"
        if force or not _done(d, "intent_interpolation.csv"):
            t0 = time.time()
            ts = interp.sweep_values(11, 0.0, 1.0)
            step = 1.0 / 10
            ts = sorted(
                set(
                    [round(-step, 6), round(-2 * step, 6)]
                    + ts
                    + [round(1.0 + step, 6), round(1.0 + 2 * step, 6)]
                )
            )
            interp.run_sweep(
                staged,
                d,
                profile_a="bandwidth_max",
                profile_b="delay_averse",
                ts=ts,
                max_pairs=max_pairs,
                hour_stride=1,
                pairs=None,
                progress=progress,
                run_context=ctx,
            )
            out["zeroshot_s"] = round(time.time() - t0, 1)
        else:
            progress("  zero-shot sweep: cached")

    if "pathcount" in studies:
        d = staged / "pathcount"
        if force or not _done(d, "pathcount_scaling.csv", "order_invariance.json"):
            t0 = time.time()
            pathcount.run(
                staged,
                d,
                profile_name="balanced_extreme",
                n_values=pathcount.DEFAULT_NS,
                max_pairs=max_pairs,
                hour_stride=1,
                n_permutations=permutations,
                pairs=None,
                progress=progress,
                run_context=ctx,
            )
            out["pathcount_s"] = round(time.time() - t0, 1)
        else:
            progress("  path-count scaling: cached")

    if "probing" in studies:
        d = staged / "probing"
        if force or not _done(d, "probing_quality.csv", "ceiling_by_congestion.csv"):
            t0 = time.time()
            # Balanced first: it drives probing_quality.csv / ceiling_by_congestion.csv,
            # which is the schema fig:p1eval:probing and fig:p1eval:ceiling read.
            profiles = ["balanced_extreme"] + [
                p for p in INTENT_PROFILES if p != "balanced_extreme"
            ]
            run_probing_ceiling(
                staged,
                d,
                max_pairs=max_pairs,
                n_congestion_bins=congestion_bins,
                progress=progress,
                run_context=ctx,
                profiles=profiles,
                agent_key=SHIPPED_AGENT,
            )
            out["probing_s"] = round(time.time() - t0, 1)
        else:
            progress("  probing/ceiling: cached")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seed directory names or numbers (default: every seed*/ present).",
    )
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument("--congestion-bins", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute studies whose outputs already exist.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    seeds_dir = run_path / "seeds"
    if args.seeds:
        seed_dirs = [
            seeds_dir / (s if s.startswith("seed") else f"seed{s}") for s in args.seeds
        ]
    else:
        seed_dirs = sorted(seeds_dir.glob("seed*"))
    if not seed_dirs:
        raise SystemExit(f"No seed directories under {seeds_dir}.")

    print(f"Per-seed results on {run_path}")
    print(f"  agent:   {SHIPPED_AGENT} (FiLM excluded by staging)")
    print(f"  seeds:   {', '.join(d.name for d in seed_dirs)}")
    print(f"  studies: {', '.join(args.studies)}")

    t0 = time.time()
    ctx = load_run_context(run_path)
    print(
        f"  run context loaded in {time.time() - t0:.1f}s "
        f"({len(ctx[3])} routable pairs; evaluating the first {args.max_pairs})"
    )

    timings: Dict[str, Any] = {}
    for seed_dir in seed_dirs:
        print(f"\n=== {seed_dir.name} ===")
        staged = stage_seed(run_path, seed_dir)
        timings[seed_dir.name] = run_seed(
            staged,
            ctx,
            max_pairs=args.max_pairs,
            studies=args.studies,
            force=args.force,
            congestion_bins=args.congestion_bins,
            permutations=args.permutations,
        )

    summary = {
        "run_dir": str(run_path),
        "agent": SHIPPED_AGENT,
        "seeds": [d.name for d in seed_dirs],
        "studies": list(args.studies),
        "max_pairs": args.max_pairs,
        "seconds_by_seed": timings,
        "total_seconds": round(time.time() - t0, 1),
    }
    out_json = seeds_dir / "seed_result_sweep.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['total_seconds']}s. Manifest: {out_json}")


if __name__ == "__main__":
    main()
