#!/usr/bin/env python3
"""Variable path sets: quality vs candidate count, and order invariance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pipeline.intent_cond_eval import (
    EVAL_HOURS,
    MAX_EVAL_PAIRS,
    METHOD_LABELS,
    _eval_pairs,
    _full_probe_cost_ms,
    compute_action_dim_from_run,
    load_conditional_agents,
    load_flat_agent,
    load_scoring_agent,
)
from src.pipeline.run_dirs import resolve_run_dir
from src.rl.reward_profiles import get_profile
from src.simulation.evaluation_env import encode_reward_weights_for_conditional
from src.simulation.run_context import load_run_context, make_env

DEFAULT_NS = [2, 4, 8, 16, 30]


def _subset_obs(obs: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {"global": obs["global"], "paths": obs["paths"][idx]}


def run(
    run_path: Path,
    out_dir: Path,
    *,
    profile_name: str,
    n_values: Sequence[int],
    max_pairs: int,
    hour_stride: int,
    n_permutations: int,
    pairs: Optional[Sequence[Tuple[int, int]]],
    progress=print,
    run_context: Optional[tuple] = None,
) -> Dict[str, Any]:
    # ``run_context`` lets a caller (the seed sweep) load the ~180 MB path store
    # once and evaluate every seed's checkpoints on the identical contexts.
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs, pairs)
    hours = EVAL_HOURS[:: max(1, hour_stride)]
    profile = get_profile(profile_name)

    agents: Dict[str, Any] = {}
    action_dim = compute_action_dim_from_run(run_path, path_store)
    flat = load_flat_agent(run_path, action_dim)
    if flat is not None:
        agents["flat_dqn"] = flat
    scoring = load_scoring_agent(run_path)
    if scoring is not None:
        agents["scoring_enhanced"] = scoring
    agents.update(load_conditional_agents(run_path))
    method_order = [
        m
        for m in (
            "flat_dqn",
            "scoring_enhanced",
            "conditional_concat",
            "conditional_concat_2stream",
            "conditional_film",
        )
        if m in agents
    ]

    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=profile.weights,
    )
    rng = np.random.default_rng(20260725)

    # (method, N) -> reward / regret accumulators
    acc: Dict[Tuple[str, int], Dict[str, List[float]]] = {
        (m, n): {"reward": [], "regret": [], "goodput": []}
        for m in method_order
        for n in n_values
    }
    # method -> permutation agreement counters
    perm_agree = {m: [0, 0] for m in method_order}  # [matches, trials]
    n_ctx = 0

    for hour_idx in hours:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            n_all = len(env.available_paths)
            if n_all < 2:
                continue
            env.reward_weights = profile.weights
            pms = env.path_metrics_snapshot()
            full_obs = env.observe_scoring()
            cond_global = np.concatenate(
                [
                    full_obs["global"],
                    encode_reward_weights_for_conditional(profile.weights),
                ]
            ).astype(np.float32)
            n_ctx += 1

            for n in n_values:
                if n > n_all:
                    continue
                # A random candidate subset, shared by every method so the
                # comparison is paired.
                idx = np.sort(rng.choice(n_all, size=n, replace=False))
                sub_pms = [pms[i] for i in idx]
                sub_bw = [float(m.get("bandwidth_mbps") or 0.0) for m in sub_pms]
                max_bw = max(sub_bw) if sub_bw else 0.0
                costs = [
                    _full_probe_cost_ms(env, m.get("hop_count", 1)) for m in sub_pms
                ]
                # Best attainable reward on this restricted set (the per-N oracle).
                best_r = max(
                    env.compute_reward(
                        m,
                        max_possible_bw=max_bw,
                        probe_cost_ms=costs[j],
                        num_probes_in_step=1,
                    )
                    for j, m in enumerate(sub_pms)
                )

                sub_obs = _subset_obs(full_obs, idx)
                sub_cond = {"global": cond_global, "paths": sub_obs["paths"]}

                for method in method_order:
                    if method == "flat_dqn":
                        mask = np.zeros(action_dim, dtype=bool)
                        mask[idx[idx < action_dim]] = True
                        if not mask.any():
                            continue
                        a_global = int(
                            agents[method].act(env.observe_flat(), action_mask=mask)
                        )
                        # Map back to a position within the subset.
                        j = (
                            int(np.where(idx == a_global)[0][0])
                            if a_global in idx
                            else 0
                        )
                    elif method == "scoring_enhanced":
                        j = int(agents[method].act(sub_obs, evaluate=True))
                    else:
                        j = int(agents[method].act(sub_cond, evaluate=True))
                    if j >= n:
                        j = 0
                    r = env.compute_reward(
                        sub_pms[j],
                        max_possible_bw=max_bw,
                        probe_cost_ms=costs[j],
                        num_probes_in_step=1,
                    )
                    a = acc[(method, n)]
                    a["reward"].append(float(r))
                    a["regret"].append(float(best_r - r))
                    a["goodput"].append(sub_bw[j])

            # --- order invariance on the full candidate set ---
            if n_permutations > 0:
                for method in method_order:
                    if method == "flat_dqn":
                        continue  # index-addressed action space; not equivariant
                    obs_m = (
                        full_obs
                        if method == "scoring_enhanced"
                        else {
                            "global": cond_global,
                            "paths": full_obs["paths"],
                        }
                    )
                    base = int(agents[method].act(obs_m, evaluate=True))
                    for _ in range(n_permutations):
                        perm = rng.permutation(n_all)
                        pobs = {
                            "global": obs_m["global"],
                            "paths": obs_m["paths"][perm],
                        }
                        pj = int(agents[method].act(pobs, evaluate=True))
                        chosen_original = int(perm[pj]) if pj < n_all else -1
                        perm_agree[method][1] += 1
                        perm_agree[method][0] += int(chosen_original == base)

        if n_ctx and n_ctx % (24 * max(1, len(eval_pairs))) == 0:
            progress(f"  pathcount: hour {hour_idx} ({n_ctx} contexts)")

    rows: List[Dict[str, Any]] = []
    for method in method_order:
        for n in n_values:
            a = acc[(method, n)]
            if not a["reward"]:
                continue
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "n_paths": n,
                    "n_decisions": len(a["reward"]),
                    "reward_mean": float(np.mean(a["reward"])),
                    "reward_std": float(np.std(a["reward"])),
                    "regret_mean": float(np.mean(a["regret"])),
                    "regret_p90": float(np.percentile(a["regret"], 90)),
                    "goodput_mean_mbps": float(np.mean(a["goodput"])),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pathcount_scaling.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    inv = {
        "run_dir": str(run_path),
        "profile": profile_name,
        "n_contexts": n_ctx,
        "permutations_per_context": n_permutations,
        "methods": {
            m: {
                "label": METHOD_LABELS.get(m, m),
                "trials": perm_agree[m][1],
                "same_path_chosen": perm_agree[m][0],
                "agreement": (
                    perm_agree[m][0] / perm_agree[m][1]
                    if perm_agree[m][1]
                    else float("nan")
                ),
            }
            for m in method_order
            if m != "flat_dqn"
        },
    }
    json_path = out_dir / "order_invariance.json"
    with open(json_path, "w") as f:
        json.dump(inv, f, indent=2)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "rows": rows,
        "invariance": inv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--profile", default="balanced_extreme")
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_NS)
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument("--hour-stride", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--pairs-json", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "gap" / "pathcount")

    pairs = None
    if args.pairs_json:
        with open(args.pairs_json) as f:
            pairs = [(int(a), int(b)) for a, b in json.load(f)["pairs"]]

    print(f"Path-count scaling + order invariance on {run_path}")
    res = run(
        run_path,
        out_dir,
        profile_name=args.profile,
        n_values=args.n_values,
        max_pairs=args.max_pairs,
        hour_stride=args.hour_stride,
        n_permutations=args.permutations,
        pairs=pairs,
    )
    print("  mean regret vs the per-N best path:")
    for row in res["rows"]:
        print(
            f"    {row['method_label']:<22} N={row['n_paths']:>2}  "
            f"regret {row['regret_mean']:.4f}  reward {row['reward_mean']:.4f}"
        )
    print("  order invariance (same path chosen under permutation):")
    for m, s in res["invariance"]["methods"].items():
        print(f"    {s['label']:<22} {s['agreement']:.4%}  ({s['trials']} trials)")
    print(f"  saved: {res['csv']}")
    print(f"  saved: {res['json']}")


if __name__ == "__main__":
    main()
