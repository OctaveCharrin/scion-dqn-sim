#!/usr/bin/env python3
"""Zero-shot generalization across intents (Chapter 4, sec:p1eval:zeroshot).

Every intent reported in the chapter is one of the six profiles the conditional
agents trained on, so the results establish only that one network can store six
behaviors -- not that it learned a mapping over the intent space. This script
presents *unseen* weight vectors at inference and asks whether behaviour moves
correspondingly.

It sweeps ``w(t) = (1-t)*w_a + t*w_b`` between two trained profiles that
genuinely conflict (default: ``bandwidth_max`` -> ``delay_averse``, i.e. maximize
throughput vs. minimize delay). ``t`` in [0, 1] interpolates; values outside that
range extrapolate beyond both trained endpoints, which the design intent also
claims. Only ``t = 0`` and ``t = 1`` are weight vectors the agent has ever seen.

For each ``t`` we log the chosen path's latency, goodput, trust, loss and hop
count, plus the reward earned under each endpoint objective. A monotone curve
through the two endpoint behaviours supports the zero-shot claim; a step function
at ``t ~ 0.5`` would show the network memorized discrete points.

Writes ``intent_interpolation.csv`` (per-t, per-method aggregates),
``intent_interpolation_summary.json`` and a figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pipeline.chapter6_eval import (
    MAX_EVAL_PAIRS,
    METHOD_LABELS,
    EVAL_HOURS,
    _eval_pairs,
    _full_probe_cost_ms,
    _intrinsic_trust,
    load_conditional_agents,
)
from src.pipeline.run_dirs import resolve_run_dir
from src.rl.reward_profiles import get_profile
from src.simulation.evaluation_env import (
    RewardWeights,
    encode_reward_weights_for_conditional,
)
from src.simulation.run_context import load_run_context, make_env


def lerp_weights(a: RewardWeights, b: RewardWeights, t: float) -> RewardWeights:
    """Affine blend of two intent vectors; ``t`` outside [0,1] extrapolates."""
    return RewardWeights(
        w1=a.w1 + t * (b.w1 - a.w1),
        w2=a.w2 + t * (b.w2 - a.w2),
        w3=a.w3 + t * (b.w3 - a.w3),
        w4=a.w4 + t * (b.w4 - a.w4),
        w_probe=a.w_probe + t * (b.w_probe - a.w_probe),
    )


def sweep_values(steps: int, t_min: float, t_max: float) -> List[float]:
    return [round(float(v), 6) for v in np.linspace(t_min, t_max, steps)]


def run_sweep(
    run_path: Path,
    out_dir: Path,
    *,
    profile_a: str,
    profile_b: str,
    ts: Sequence[float],
    max_pairs: int,
    hour_stride: int,
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

    agents = load_conditional_agents(run_path)
    if not agents:
        raise FileNotFoundError(f"No conditional checkpoints under {run_path}.")
    method_order = [
        m
        for m in (
            "conditional_film",
            "conditional_concat",
            "conditional_concat_2stream",
        )
        if m in agents
    ]

    wa = get_profile(profile_a).weights
    wb = get_profile(profile_b).weights
    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=RewardWeights(),
    )

    # (method, t) -> accumulators
    acc: Dict[Tuple[str, float], Dict[str, List[float]]] = {
        (m, t): {
            "latency": [],
            "goodput": [],
            "trust": [],
            "loss": [],
            "hops": [],
            "reward_at_t": [],
            "reward_a": [],
            "reward_b": [],
            "choice": [],
        }
        for m in method_order
        for t in ts
    }

    n_contexts = 0
    for hour_idx in hours:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            n = len(env.available_paths)
            if n == 0:
                continue
            pms = env.path_metrics_snapshot()
            max_bw = max(
                (float(m.get("bandwidth_mbps") or 0.0) for m in pms), default=0.0
            )
            probe_costs = [_full_probe_cost_ms(env, m.get("hop_count", 1)) for m in pms]
            n_contexts += 1

            for t in ts:
                w = lerp_weights(wa, wb, t)
                env.reward_weights = w
                obs = env.observe_scoring()
                cond_obs = {
                    "global": np.concatenate(
                        [obs["global"], encode_reward_weights_for_conditional(w)]
                    ).astype(np.float32),
                    "paths": obs["paths"],
                }
                for method in method_order:
                    action = int(agents[method].act(cond_obs, evaluate=True))
                    if action >= n:
                        action = 0
                    pm = pms[action]
                    a = acc[(method, t)]
                    lat = float(pm.get("latency_ms", 50.0))
                    loss = float(pm.get("loss_rate", 0.0))
                    a["latency"].append(lat)
                    a["goodput"].append(float(pm.get("bandwidth_mbps") or 0.0))
                    a["trust"].append(_intrinsic_trust(lat, loss))
                    a["loss"].append(loss)
                    a["hops"].append(float(pm.get("hop_count", 1)))
                    a["choice"].append(float(action))
                    a["reward_at_t"].append(
                        env.compute_reward(
                            pm,
                            max_possible_bw=max_bw,
                            probe_cost_ms=probe_costs[action],
                            num_probes_in_step=1,
                        )
                    )
                    for key, wref in (("reward_a", wa), ("reward_b", wb)):
                        a[key].append(
                            env.compute_reward(
                                pm,
                                max_possible_bw=max_bw,
                                probe_cost_ms=probe_costs[action],
                                num_probes_in_step=1,
                                weights=wref.to_dict(),
                            )
                        )
        if n_contexts and n_contexts % (24 * max(1, len(eval_pairs))) == 0:
            progress(f"  interpolation: hour {hour_idx} ({n_contexts} contexts)")

    rows: List[Dict[str, Any]] = []
    for method in method_order:
        for t in ts:
            a = acc[(method, t)]
            w = lerp_weights(wa, wb, t)
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "t": t,
                    "trained_endpoint": int(t in (0.0, 1.0)),
                    "w1": w.w1,
                    "w2": w.w2,
                    "w3": w.w3,
                    "w4": w.w4,
                    "w_probe": w.w_probe,
                    "n": len(a["latency"]),
                    "latency_mean_ms": float(np.mean(a["latency"])),
                    "latency_p50_ms": float(np.percentile(a["latency"], 50)),
                    "goodput_mean_mbps": float(np.mean(a["goodput"])),
                    "trust_mean": float(np.mean(a["trust"])),
                    "loss_mean": float(np.mean(a["loss"])),
                    "hops_mean": float(np.mean(a["hops"])),
                    "reward_at_t_mean": float(np.mean(a["reward_at_t"])),
                    "reward_under_a_mean": float(np.mean(a["reward_a"])),
                    "reward_under_b_mean": float(np.mean(a["reward_b"])),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "intent_interpolation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Monotonicity: does chosen latency move steadily from endpoint a to b?
    summary: Dict[str, Any] = {
        "run_dir": str(run_path),
        "profile_a": profile_a,
        "profile_b": profile_b,
        "t_values": list(ts),
        "n_contexts": n_contexts,
        "n_pairs": len(eval_pairs),
        "n_hours": len(hours),
        "methods": {},
    }
    inner = [t for t in ts if 0.0 <= t <= 1.0]
    for method in method_order:
        lat = np.array(
            [float(np.mean(acc[(method, t)]["latency"])) for t in inner], dtype=float
        )
        diffs = np.diff(lat)
        # Spearman rank correlation of chosen latency against t.
        from scipy.stats import spearmanr

        rho, pval = spearmanr(inner, lat)
        # How much of the endpoint-to-endpoint change happens at interior t?
        total = abs(lat[-1] - lat[0])
        largest_step = float(np.max(np.abs(diffs))) if diffs.size else 0.0
        summary["methods"][method] = {
            "label": METHOD_LABELS.get(method, method),
            "latency_at_t0_ms": float(lat[0]),
            "latency_at_t1_ms": float(lat[-1]),
            "latency_span_ms": float(lat[0] - lat[-1]),
            "monotone_decreasing": bool(np.all(diffs <= 1e-9)),
            "spearman_rho_latency_vs_t": float(rho),
            "spearman_p": float(pval),
            "largest_single_step_ms": largest_step,
            "largest_step_fraction_of_span": (
                float(largest_step / total) if total > 1e-12 else float("nan")
            ),
            "goodput_at_t0_mbps": float(np.mean(acc[(method, inner[0])]["goodput"])),
            "goodput_at_t1_mbps": float(np.mean(acc[(method, inner[-1])]["goodput"])),
        }
        # Extrapolation, if requested.
        extra_lo = [t for t in ts if t < 0.0]
        extra_hi = [t for t in ts if t > 1.0]
        if extra_lo or extra_hi:
            summary["methods"][method]["extrapolation"] = {
                "latency_below_t0_ms": {
                    str(t): float(np.mean(acc[(method, t)]["latency"]))
                    for t in extra_lo
                },
                "latency_above_t1_ms": {
                    str(t): float(np.mean(acc[(method, t)]["latency"]))
                    for t in extra_hi
                },
            }

    json_path = out_dir / "intent_interpolation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--profile-a", default="bandwidth_max")
    parser.add_argument("--profile-b", default="delay_averse")
    parser.add_argument("--steps", type=int, default=11, help="Points in [0,1].")
    parser.add_argument(
        "--extrapolate",
        type=float,
        default=0.25,
        help="Also evaluate this far beyond each endpoint (0 to disable).",
    )
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument(
        "--hour-stride", type=int, default=1, help="Evaluate every Nth held-out hour."
    )
    parser.add_argument("--pairs-json", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    out_dir = args.out_dir or (run_path / "gap" / "zeroshot")

    pairs = None
    if args.pairs_json:
        with open(args.pairs_json) as f:
            pairs = [(int(a), int(b)) for a, b in json.load(f)["pairs"]]

    ts = sweep_values(args.steps, 0.0, 1.0)
    if args.extrapolate > 0:
        step = 1.0 / max(1, args.steps - 1)
        n_extra = max(1, int(round(args.extrapolate / step)))
        lo = [round(-step * k, 6) for k in range(1, n_extra + 1)]
        hi = [round(1.0 + step * k, 6) for k in range(1, n_extra + 1)]
        ts = sorted(set(lo + ts + hi))

    print(f"Intent interpolation {args.profile_a} -> {args.profile_b} on {run_path}")
    print(f"  t values: {ts}")
    result = run_sweep(
        run_path,
        out_dir,
        profile_a=args.profile_a,
        profile_b=args.profile_b,
        ts=ts,
        max_pairs=args.max_pairs,
        hour_stride=args.hour_stride,
        pairs=pairs,
    )
    for method, s in result["summary"]["methods"].items():
        print(
            f"  {s['label']:<22} latency {s['latency_at_t0_ms']:.2f} -> "
            f"{s['latency_at_t1_ms']:.2f} ms  "
            f"monotone={s['monotone_decreasing']}  rho={s['spearman_rho_latency_vs_t']:+.3f}  "
            f"largest step = {s['largest_step_fraction_of_span']:.1%} of span"
        )
    print(f"  saved: {result['csv']}")
    print(f"  saved: {result['json']}")


if __name__ == "__main__":
    main()
