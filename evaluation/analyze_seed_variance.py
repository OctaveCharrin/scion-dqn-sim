#!/usr/bin/env python3
"""Seed variance and paired significance for the conditioning ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from src.pipeline.run_dirs import resolve_run_dir


def _ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return {
            "mean": mean,
            "std": float("nan"),
            "ci95_lo": mean,
            "ci95_hi": mean,
            "n": n,
        }
    sd = float(arr.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {
        "mean": mean,
        "std": sd,
        "ci95_lo": mean - half,
        "ci95_hi": mean + half,
        "ci95_halfwidth": half,
        "n": n,
    }


def collect_seed_variance(seeds_dir: Path) -> Dict[str, Any]:
    reward: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    diverg: Dict[str, List[float]] = defaultdict(list)
    entropy: Dict[str, List[float]] = defaultdict(list)
    labels: Dict[str, str] = {}
    seeds_used: List[str] = []

    for seed_dir in sorted(seeds_dir.glob("seed*")):
        rmat = seed_dir / "ablation" / "ablation_reward_matrix.csv"
        dmat = seed_dir / "ablation" / "ablation_behavioral_divergence.csv"
        if not rmat.is_file() or not dmat.is_file():
            continue
        seeds_used.append(seed_dir.name)
        with open(rmat, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["method"]] = row["method_label"]
                reward[row["method"]][row["profile"]].append(float(row["reward_mean"]))
        with open(dmat, newline="") as f:
            for row in csv.DictReader(f):
                diverg[row["method"]].append(float(row["behavioral_divergence"]))
                entropy[row["method"]].append(float(row["choice_entropy_mean"]))

    return {
        "seeds": seeds_used,
        "n_seeds": len(seeds_used),
        "reward_by_method_profile": {
            m: {p: _ci95(v) for p, v in profs.items()} for m, profs in reward.items()
        },
        "adaptivity_by_method": {m: _ci95(v) for m, v in diverg.items()},
        "choice_entropy_by_method": {m: _ci95(v) for m, v in entropy.items()},
        "method_labels": labels,
    }


def paired_tests(npz_path: Path, max_pairs_of_methods: int = 64) -> Dict[str, Any]:
    with np.load(npz_path) as npz:
        data = {k: npz[k].astype(np.float64) for k in npz.files}
    by_profile: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for key, arr in data.items():
        method, profile = key.split("|", 1)
        by_profile[profile][method] = arr

    out: Dict[str, Any] = {"source": str(npz_path), "comparisons": []}
    for profile, methods in sorted(by_profile.items()):
        names = sorted(methods)
        for a, b in list(combinations(names, 2))[:max_pairs_of_methods]:
            xa, xb = methods[a], methods[b]
            if xa.shape != xb.shape or xa.size == 0:
                continue
            diff = xa - xb
            n_nonzero = int(np.count_nonzero(diff))
            if n_nonzero == 0:
                res = {"statistic": float("nan"), "pvalue": 1.0}
            else:
                w = stats.wilcoxon(
                    xa, xb, zero_method="zsplit", alternative="two-sided"
                )
                res = {"statistic": float(w.statistic), "pvalue": float(w.pvalue)}
            # Common-language effect size: P(a > b) among contexts where they differ.
            wins = int(np.sum(diff > 0))
            losses = int(np.sum(diff < 0))
            out["comparisons"].append(
                {
                    "profile": profile,
                    "method_a": a,
                    "method_b": b,
                    "n_contexts": int(xa.size),
                    "n_differing": n_nonzero,
                    "mean_a": float(xa.mean()),
                    "mean_b": float(xb.mean()),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(np.median(diff)),
                    "a_better_count": wins,
                    "b_better_count": losses,
                    "wilcoxon_statistic": res["statistic"],
                    "wilcoxon_p": res["pvalue"],
                    "significant_at_0.05": bool(res["pvalue"] < 0.05),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--seeds-dir", type=Path, default=None)
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="ablation_per_context_rewards.npz for the paired test "
        "(default: <run>/gap/ablation_main/...).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    seeds_dir = args.seeds_dir or (run_path / "seeds")
    npz_path = args.npz or (
        run_path / "gap" / "ablation_main" / "ablation_per_context_rewards.npz"
    )
    out_dir = args.out_dir or (run_path / "gap" / "stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    if seeds_dir.is_dir():
        sv = collect_seed_variance(seeds_dir)
        with open(out_dir / "seed_variance.json", "w") as f:
            json.dump(sv, f, indent=2)
        print(f"Seed variance over {sv['n_seeds']} seeds ({', '.join(sv['seeds'])})")
        print(f"{'method':<28} {'adaptivity mean':>16} {'95% CI':>22}")
        for m, s in sv["adaptivity_by_method"].items():
            label = sv["method_labels"].get(m, m)
            print(
                f"{label:<28} {s['mean']:>16.4f} "
                f"  [{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]"
            )
        print()
        for profile in ("delay_averse", "bandwidth_max"):
            print(f"  reward under {profile}:")
            for m, profs in sv["reward_by_method_profile"].items():
                if profile in profs:
                    s = profs[profile]
                    label = sv["method_labels"].get(m, m)
                    print(
                        f"    {label:<26} {s['mean']:.4f}  "
                        f"[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]"
                    )
        print(f"  saved: {out_dir / 'seed_variance.json'}")
    else:
        print(f"(no seeds dir at {seeds_dir}; skipping across-seed aggregation)")

    if npz_path.is_file():
        sig = paired_tests(npz_path)
        with open(out_dir / "significance.json", "w") as f:
            json.dump(sig, f, indent=2)
        print(f"\nPaired Wilcoxon over {npz_path}")
        # stdout shows comparisons involving the shipped agent; the JSON has all pairs.
        shipped = "conditional_concat_2stream"
        for c in sig["comparisons"]:
            if shipped not in (c["method_a"], c["method_b"]):
                continue
            flag = "*" if c["significant_at_0.05"] else " "
            print(
                f"  {c['profile']:<18} {c['method_a']:<28} vs {c['method_b']:<28} "
                f"diff {c['mean_diff']:+.5f}  p={c['wilcoxon_p']:.3g}{flag}"
            )
        print(f"  saved: {out_dir / 'significance.json'}")
    else:
        print(f"(no npz at {npz_path}; skipping paired tests)")


if __name__ == "__main__":
    main()
