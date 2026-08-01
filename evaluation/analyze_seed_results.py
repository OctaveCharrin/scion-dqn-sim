#!/usr/bin/env python3
"""Aggregate the per-seed Chapter 4 studies and test whether the chapter's claims
survive five training seeds.

``run_seed_result_sweep.py`` produces ``seeds/seed<N>/shipped/{intent,zeroshot,
pathcount,probing}/``; this reduces those to means with 95% confidence intervals
over seeds, using the same interval as ``analyze_seed_variance.py`` so every
seeded number in the chapter is computed one way.

It also *checks the claims*, because several of them are quantitatively sharp and
a mean alone does not say whether they hold: the intent-alignment diagonal must
win all four columns, the zero-shot Spearman must be exactly -1.000 with no
single step carrying more than 24.7% of the span, order invariance must be
exactly 100%, the learned selector must beat every heuristic on three of four
intents and stay within 0.0015 of lowest-latency on the fourth, and the ceiling
must descend ~24% while reward stays flat. Each is evaluated per seed and
reported as survives / fails with the spread it was tested against.

Writes ``<run>/seeds/aggregate/`` (aggregate CSVs the figures read, plus
``claims.json``) and prints a report.

Usage: uv run python analyze_seed_results.py <run_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analyze_seed_variance import _ci95
from src.pipeline.chapter6_eval import INTENT_LABELS, INTENT_PROFILES, METHOD_LABELS
from src.pipeline.run_dirs import resolve_run_dir

SHIPPED_AGENT = "conditional_concat_2stream"
HEURISTICS = [
    "shortest_path",
    "widest_path",
    "lowest_latency",
    "ecmp",
    "scion_default",
    "random",
]


# ------------------------------------------------------------------ io helpers
def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def seed_dirs(run_path: Path, names: Optional[Sequence[str]] = None) -> List[Path]:
    root = run_path / "seeds"
    dirs = (
        [root / (n if n.startswith("seed") else f"seed{n}") for n in names]
        if names
        else sorted(root.glob("seed*"))
    )
    return [d for d in dirs if (d / "shipped").is_dir()]


def aggregate_rows(
    per_seed: Dict[str, List[Dict[str, str]]],
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    passthrough: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Group identically-keyed rows across seeds into mean / CI / per-seed values.

    Every aggregate CSV carries ``<col>_seedvals`` alongside the interval so the
    figures can draw per-seed traces from the same file the text is written from
    -- the spread is never re-derived separately for the picture.
    """
    grouped: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    order: List[Tuple[str, ...]] = []
    for seed, rows in sorted(per_seed.items()):
        for row in rows:
            key = tuple(row[c] for c in key_cols)
            if key not in grouped:
                grouped[key] = {
                    "_keys": key,
                    "_pass": {c: row.get(c, "") for c in passthrough},
                    "_vals": defaultdict(list),
                    "_seeds": [],
                }
                order.append(key)
            grouped[key]["_seeds"].append(seed)
            for col in value_cols:
                grouped[key]["_vals"][col].append(_f(row, col))

    out: List[Dict[str, Any]] = []
    for key in order:
        g = grouped[key]
        rec: Dict[str, Any] = dict(zip(key_cols, key))
        rec.update(g["_pass"])
        rec["n_seeds"] = len(g["_seeds"])
        rec["seeds"] = ";".join(g["_seeds"])
        for col in value_cols:
            vals = g["_vals"][col]
            s = _ci95(vals)
            rec[f"{col}_mean"] = s["mean"]
            rec[f"{col}_sd"] = s["std"]
            rec[f"{col}_ci_lo"] = s["ci95_lo"]
            rec[f"{col}_ci_hi"] = s["ci95_hi"]
            rec[f"{col}_seedvals"] = ";".join(f"{v:.10g}" for v in vals)
        out.append(rec)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _fmt(s: Dict[str, float], digits: int = 4) -> str:
    return (
        f"{s['mean']:.{digits}f} [{s['ci95_lo']:.{digits}f}, {s['ci95_hi']:.{digits}f}]"
    )


def _agg(values: Iterable[float]) -> Dict[str, float]:
    return _ci95([float(v) for v in values])


# --------------------------------------------------------------- 1. intent matrix
def study_intent_matrix(dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    per_seed = {
        d.name: _read_csv(d / "shipped" / "intent" / "intent_reward_matrix.csv")
        for d in dirs
    }
    rows = aggregate_rows(
        per_seed,
        ["intent_told", "intent_scored"],
        ["reward_mean"],
        passthrough=["intent_told_label", "intent_scored_label"],
    )
    write_csv(out_dir / "intent_reward_matrix_seeds.csv", rows)

    # Per-seed matrices, so "does the diagonal win this column" is a per-seed test.
    mats: Dict[str, np.ndarray] = {}
    for seed, srows in per_seed.items():
        m = np.full((len(INTENT_PROFILES), len(INTENT_PROFILES)), np.nan)
        idx = {n: i for i, n in enumerate(INTENT_PROFILES)}
        for r in srows:
            m[idx[r["intent_told"]], idx[r["intent_scored"]]] = _f(r, "reward_mean")
        mats[seed] = m

    columns: Dict[str, Any] = {}
    for j, scored in enumerate(INTENT_PROFILES):
        margins, ranges, wins = [], [], []
        runner_up: Dict[str, int] = defaultdict(int)
        for seed, m in mats.items():
            col = m[:, j]
            diag = col[j]
            off = np.delete(col, j)
            best_off = float(off.max())
            best_i = int(np.delete(np.arange(len(col)), j)[int(np.argmax(off))])
            margins.append(diag - best_off)
            ranges.append(float(col.max() - col.min()))
            wins.append(bool(diag > best_off))
            runner_up[INTENT_PROFILES[best_i]] += 1
        margin = _agg(margins)
        columns[scored] = {
            "label": INTENT_LABELS[scored],
            "diagonal_wins_in_seeds": int(sum(wins)),
            "n_seeds": len(wins),
            "wins_all_seeds": all(wins),
            # A win is only meaningful if the margin's interval clears zero.
            "margin_over_best_off_diagonal": margin,
            "margin_ci_excludes_zero": bool(margin["ci95_lo"] > 0),
            "column_range": _agg(ranges),
            "nearest_rival": max(runner_up, key=runner_up.get),
        }
    return {"columns": columns, "n_seeds": len(mats)}


# ------------------------------------------------------- 2. per-intent selections
def study_intent_metrics(dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    """Chosen-path metric distributions per conditioning intent.

    The raw CSV is one row per selection (43008 per seed), so we reduce each seed
    to its per-intent summary first and aggregate those -- the confidence interval
    is then over seeds, not over selections, which is the whole point.
    """
    per_seed_summary: Dict[str, List[Dict[str, str]]] = {}
    pooled: Dict[str, Dict[str, List[float]]] = {
        col: {name: [] for name in INTENT_PROFILES}
        for col in ("latency_ms", "bandwidth_mbps", "trust", "loss_rate")
    }
    for d in dirs:
        rows = _read_csv(d / "shipped" / "intent" / "intent_selection_metrics.csv")
        by_intent: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in rows:
            intent = r["intent"]
            for col in ("latency_ms", "bandwidth_mbps", "trust", "loss_rate"):
                v = _f(r, col)
                by_intent[intent][col].append(v)
                pooled[col][intent].append(v)
        summary = []
        for intent in INTENT_PROFILES:
            vals = by_intent[intent]
            loss = np.asarray(vals["loss_rate"], dtype=float)
            summary.append(
                {
                    "intent": intent,
                    "intent_label": INTENT_LABELS[intent],
                    "n_selections": str(len(loss)),
                    "latency_mean_ms": str(float(np.mean(vals["latency_ms"]))),
                    "latency_p50_ms": str(float(np.percentile(vals["latency_ms"], 50))),
                    "goodput_mean_mbps": str(float(np.mean(vals["bandwidth_mbps"]))),
                    "trust_mean": str(float(np.mean(vals["trust"]))),
                    "loss_exposure_pct": str(100.0 * float(np.mean(loss > 0.0))),
                }
            )
        per_seed_summary[d.name] = summary

    value_cols = [
        "latency_mean_ms",
        "latency_p50_ms",
        "goodput_mean_mbps",
        "trust_mean",
        "loss_exposure_pct",
    ]
    rows = aggregate_rows(
        per_seed_summary, ["intent"], value_cols, passthrough=["intent_label"]
    )
    write_csv(out_dir / "intent_selection_summary_seeds.csv", rows)

    # Pooled per-selection distributions, for the boxplot bodies.
    box_rows = []
    for intent in INTENT_PROFILES:
        for col in ("latency_ms", "bandwidth_mbps", "trust"):
            arr = np.asarray(pooled[col][intent], dtype=float)
            q = np.percentile(arr, [5, 25, 50, 75, 95])
            box_rows.append(
                {
                    "intent": intent,
                    "intent_label": INTENT_LABELS[intent],
                    "metric": col,
                    "n": arr.size,
                    "p5": q[0],
                    "q1": q[1],
                    "median": q[2],
                    "q3": q[3],
                    "p95": q[4],
                    "mean": float(arr.mean()),
                }
            )
    write_csv(out_dir / "intent_selection_boxstats_seeds.csv", box_rows)

    lookup = {r["intent"]: r for r in rows}
    latency_gap = _agg(
        np.asarray(
            lookup["bandwidth_max"]["latency_mean_ms_seedvals"].split(";"), float
        )
        - np.asarray(
            lookup["delay_averse"]["latency_mean_ms_seedvals"].split(";"), float
        )
    )
    trust_gap = _agg(
        np.asarray(lookup["delay_averse"]["trust_mean_seedvals"].split(";"), float)
        - np.asarray(lookup["loss_averse"]["trust_mean_seedvals"].split(";"), float)
    )
    return {
        "per_intent": {
            r["intent"]: {
                k: {
                    "mean": r[f"{k}_mean"],
                    "ci95_lo": r[f"{k}_ci_lo"],
                    "ci95_hi": r[f"{k}_ci_hi"],
                }
                for k in value_cols
            }
            for r in rows
        },
        "latency_saving_lowlat_vs_thpt_ms": latency_gap,
        "trust_lowlat_minus_lowloss": trust_gap,
        "trust_lowlat_highest_in_all_seeds": bool(trust_gap["ci95_lo"] > 0),
    }


# -------------------------------------------------------------------- 3. zeroshot
def study_zeroshot(dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    per_seed = {
        d.name: _read_csv(d / "shipped" / "zeroshot" / "intent_interpolation.csv")
        for d in dirs
    }
    rows = aggregate_rows(
        per_seed,
        ["method", "t"],
        ["latency_mean_ms", "goodput_mean_mbps", "trust_mean", "reward_at_t_mean"],
        passthrough=["method_label", "trained_endpoint"],
    )
    write_csv(out_dir / "intent_interpolation_seeds.csv", rows)

    summaries = {
        d.name: json.loads(
            (
                d / "shipped" / "zeroshot" / "intent_interpolation_summary.json"
            ).read_text()
        )
        for d in dirs
    }
    out: Dict[str, Any] = {"methods": {}}
    for method in (SHIPPED_AGENT, "conditional_concat"):
        rhos, steps, spans, t0s, t1s, mono = [], [], [], [], [], []
        for seed, s in summaries.items():
            m = s["methods"].get(method)
            if not m:
                continue
            rhos.append(m["spearman_rho_latency_vs_t"])
            steps.append(m["largest_step_fraction_of_span"])
            spans.append(m["latency_span_ms"])
            t0s.append(m["latency_at_t0_ms"])
            t1s.append(m["latency_at_t1_ms"])
            mono.append(bool(m["monotone_decreasing"]))
        if not rhos:
            continue
        extrap = []
        for seed, srows in per_seed.items():
            hits = [
                _f(r, "latency_mean_ms")
                for r in srows
                if r["method"] == method and abs(_f(r, "t") - 1.2) < 1e-9
            ]
            if hits:
                extrap.append(hits[0])
        out["methods"][method] = {
            "label": METHOD_LABELS[method],
            "spearman_rho": {"min": min(rhos), "max": max(rhos), "per_seed": rhos},
            "rho_exactly_minus_one_in_all_seeds": all(r <= -0.999999 for r in rhos),
            "monotone_in_all_seeds": all(mono),
            "largest_step_fraction": {
                "min": min(steps),
                "max": max(steps),
                "per_seed": steps,
                **_agg(steps),
            },
            "largest_step_under_24_7pct_in_all_seeds": all(s <= 0.247 for s in steps),
            "latency_at_t0_ms": _agg(t0s),
            "latency_at_t1_ms": _agg(t1s),
            "latency_span_ms": _agg(spans),
            "latency_at_t1_2_ms": _agg(extrap) if extrap else None,
        }
    if SHIPPED_AGENT in out["methods"] and "conditional_concat" in out["methods"]:
        a = out["methods"][SHIPPED_AGENT]["latency_span_ms"]["mean"]
        b = out["methods"]["conditional_concat"]["latency_span_ms"]["mean"]
        out["span_ratio_2stream_over_valueconcat"] = float(a / b) if b else float("nan")
    return out


# ------------------------------------------------------------------- 4. pathcount
def study_pathcount(dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    per_seed = {
        d.name: _read_csv(d / "shipped" / "pathcount" / "pathcount_scaling.csv")
        for d in dirs
    }
    rows = aggregate_rows(
        per_seed,
        ["method", "n_paths"],
        ["regret_mean", "reward_mean", "goodput_mean_mbps"],
        passthrough=["method_label"],
    )
    write_csv(out_dir / "pathcount_scaling_seeds.csv", rows)

    invariance: Dict[str, Any] = {}
    for d in dirs:
        inv = json.loads(
            (d / "shipped" / "pathcount" / "order_invariance.json").read_text()
        )
        for method, s in inv["methods"].items():
            invariance.setdefault(
                method,
                {"label": s["label"], "trials": [], "agreement": [], "misses": []},
            )
            invariance[method]["trials"].append(s["trials"])
            invariance[method]["agreement"].append(s["agreement"])
            invariance[method]["misses"].append(s["trials"] - s["same_path_chosen"])

    inv_out = {
        m: {
            "label": v["label"],
            "trials_per_seed": v["trials"],
            "total_trials": int(sum(v["trials"])),
            "agreement_per_seed": v["agreement"],
            "mismatches_per_seed": v["misses"],
            "exactly_100pct_in_all_seeds": all(a >= 1.0 for a in v["agreement"]),
            "min_agreement": min(v["agreement"]),
        }
        for m, v in invariance.items()
    }

    # Scoring agents vs the flat DQN across the whole N range.
    scoring = [r for r in rows if r["method"] != "flat_dqn"]
    flat = [r for r in rows if r["method"] == "flat_dqn"]
    return {
        "order_invariance": inv_out,
        "scoring_regret_range": (
            [
                min(r["regret_mean_mean"] for r in scoring),
                max(r["regret_mean_mean"] for r in scoring),
            ]
            if scoring
            else None
        ),
        "flat_regret_range": (
            [
                min(r["regret_mean_mean"] for r in flat),
                max(r["regret_mean_mean"] for r in flat),
            ]
            if flat
            else None
        ),
    }


# --------------------------------------------------------------------- 5. probing
def study_probing(dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    per_seed = {
        d.name: _read_csv(d / "shipped" / "probing" / "probing_quality.csv")
        for d in dirs
    }
    rows = aggregate_rows(
        per_seed,
        ["method"],
        [
            "probe_cost_per_selection_ms",
            "probes_per_selection",
            "goodput_mean_mbps",
            "reward_mean",
        ],
        passthrough=["method_label"],
    )
    write_csv(out_dir / "probing_quality_seeds.csv", rows)

    per_seed_intent = {
        d.name: _read_csv(d / "shipped" / "probing" / "probing_quality_by_intent.csv")
        for d in dirs
    }
    rows_intent = aggregate_rows(
        per_seed_intent,
        ["profile", "method"],
        [
            "probe_cost_per_selection_ms",
            "probes_per_selection",
            "goodput_mean_mbps",
            "reward_mean",
        ],
        passthrough=["profile_label", "method_label"],
    )
    write_csv(out_dir / "probing_quality_by_intent_seeds.csv", rows_intent)

    # Per intent: is the learned selector at the top, and by how much?
    per_intent: Dict[str, Any] = {}
    for profile in INTENT_PROFILES:
        # method -> seed-ordered rewards, so the margin is a paired per-seed quantity.
        rewards: Dict[str, List[float]] = defaultdict(list)
        for seed in sorted(per_seed_intent):
            for r in per_seed_intent[seed]:
                if r["profile"] == profile:
                    rewards[r["method"]].append(_f(r, "reward_mean"))
        learned = np.asarray(rewards[SHIPPED_AGENT], dtype=float)
        best_h_name, best_h_vals = None, None
        for h in HEURISTICS:
            vals = np.asarray(rewards[h], dtype=float)
            if best_h_vals is None or vals.mean() > best_h_vals.mean():
                best_h_name, best_h_vals = h, vals
        margin = _agg(learned - best_h_vals)
        per_intent[profile] = {
            "label": INTENT_LABELS[profile],
            "learned_reward": _agg(learned),
            "best_heuristic": best_h_name,
            "best_heuristic_label": METHOD_LABELS[best_h_name],
            "best_heuristic_reward": _agg(best_h_vals),
            "margin_learned_minus_best_heuristic": margin,
            "learned_wins_in_seeds": int(np.sum(learned > best_h_vals)),
            "n_seeds": int(learned.size),
            "wins_all_seeds": bool(np.all(learned > best_h_vals)),
            "margin_ci_excludes_zero": bool(margin["ci95_lo"] > 0),
        }
    n_beaten = sum(1 for v in per_intent.values() if v["wins_all_seeds"])

    by_method = {r["method"]: r for r in rows}
    learned_g = np.asarray(
        by_method[SHIPPED_AGENT]["goodput_mean_mbps_seedvals"].split(";"), float
    )
    widest_g = np.asarray(
        by_method["widest_path"]["goodput_mean_mbps_seedvals"].split(";"), float
    )
    learned_c = np.asarray(
        by_method[SHIPPED_AGENT]["probe_cost_per_selection_ms_seedvals"].split(";"),
        float,
    )
    widest_c = float(by_method["widest_path"]["probe_cost_per_selection_ms_mean"])
    return {
        "by_method": {
            r["method"]: {
                "label": r["method_label"],
                "probe_cost_ms": {
                    "mean": r["probe_cost_per_selection_ms_mean"],
                    "ci95_lo": r["probe_cost_per_selection_ms_ci_lo"],
                    "ci95_hi": r["probe_cost_per_selection_ms_ci_hi"],
                },
                "goodput_mbps": {
                    "mean": r["goodput_mean_mbps_mean"],
                    "ci95_lo": r["goodput_mean_mbps_ci_lo"],
                    "ci95_hi": r["goodput_mean_mbps_ci_hi"],
                },
                "reward": {
                    "mean": r["reward_mean_mean"],
                    "ci95_lo": r["reward_mean_ci_lo"],
                    "ci95_hi": r["reward_mean_ci_hi"],
                },
            }
            for r in rows
        },
        "per_intent": per_intent,
        "intents_beating_every_heuristic": n_beaten,
        "goodput_vs_widest_pct": _agg(100.0 * (learned_g - widest_g) / widest_g),
        "probe_cost_reduction_vs_widest": _agg(widest_c / learned_c),
    }


# --------------------------------------------------------------------- 6. ceiling
def study_ceiling(dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    per_seed = {
        d.name: _read_csv(d / "shipped" / "probing" / "ceiling_by_congestion.csv")
        for d in dirs
    }
    rows = aggregate_rows(
        per_seed,
        ["method", "congestion_bin"],
        ["congestion_mid", "goodput_mean_mbps", "reward_mean"],
        passthrough=["method_label", "n"],
    )
    write_csv(out_dir / "ceiling_by_congestion_seeds.csv", rows)

    bins = sorted({int(r["congestion_bin"]) for r in rows})
    lo_bin, hi_bin = str(bins[0]), str(bins[-1])

    def _series(method: str, bin_id: str, col: str) -> np.ndarray:
        for r in rows:
            if r["method"] == method and r["congestion_bin"] == bin_id:
                return np.asarray(r[f"{col}_seedvals"].split(";"), dtype=float)
        return np.asarray([], dtype=float)

    out: Dict[str, Any] = {"n_bins": len(bins), "methods": {}}
    for method in (SHIPPED_AGENT, "widest_path", "shortest_path", "random"):
        g_lo, g_hi = _series(method, lo_bin, "goodput_mean_mbps"), _series(
            method, hi_bin, "goodput_mean_mbps"
        )
        r_lo, r_hi = _series(method, lo_bin, "reward_mean"), _series(
            method, hi_bin, "reward_mean"
        )
        if g_lo.size == 0:
            continue
        out["methods"][method] = {
            "label": METHOD_LABELS[method],
            "goodput_light_mbps": _agg(g_lo),
            "goodput_heavy_mbps": _agg(g_hi),
            "goodput_drop_pct": _agg(100.0 * (g_lo - g_hi) / g_lo),
            "reward_light": _agg(r_lo),
            "reward_heavy": _agg(r_hi),
            "reward_drop": _agg(r_lo - r_hi),
        }
    return out


# ------------------------------------------------------------------------ report
def _claim(name: str, survives: bool, detail: str) -> Dict[str, Any]:
    return {"claim": name, "survives": survives, "detail": detail}


def build_claims(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []

    cols = res["intent_matrix"]["columns"]
    won = [c for c in cols.values() if c["wins_all_seeds"]]
    sig = [c for c in cols.values() if c["margin_ci_excludes_zero"]]
    claims.append(
        _claim(
            "Intent-alignment diagonal wins all four columns",
            len(won) == 4,
            f"diagonal wins in every seed on {len(won)}/4 columns; the margin's 95% CI "
            f"clears zero on {len(sig)}/4. Weakest: "
            + ", ".join(
                f"{c['label']} margin {_fmt(c['margin_over_best_off_diagonal'], 4)}"
                for c in sorted(
                    cols.values(),
                    key=lambda c: c["margin_over_best_off_diagonal"]["mean"],
                )[:2]
            ),
        )
    )

    z = res["zeroshot"]["methods"].get(SHIPPED_AGENT, {})
    if z:
        claims.append(
            _claim(
                "Zero-shot Spearman rho = -1.000",
                bool(z["rho_exactly_minus_one_in_all_seeds"]),
                f"per-seed rho in [{z['spearman_rho']['min']:.4f}, "
                f"{z['spearman_rho']['max']:.4f}]; monotone in all seeds: "
                f"{z['monotone_in_all_seeds']}",
            )
        )
        st = z["largest_step_fraction"]
        claims.append(
            _claim(
                "No single interpolation step carries >24.7% of the span",
                bool(z["largest_step_under_24_7pct_in_all_seeds"]),
                f"largest step per seed in [{st['min']:.1%}, {st['max']:.1%}], "
                f"mean {st['mean']:.1%} [{st['ci95_lo']:.1%}, {st['ci95_hi']:.1%}]",
            )
        )

    inv = res["pathcount"]["order_invariance"]
    exact = all(v["exactly_100pct_in_all_seeds"] for v in inv.values())
    total = sum(v["total_trials"] for v in inv.values())
    claims.append(
        _claim(
            "Order invariance is exactly 100.0000%",
            exact,
            f"{total} permutation trials across seeds and agents; minimum agreement "
            + ", ".join(f"{v['label']} {v['min_agreement']:.6%}" for v in inv.values()),
        )
    )

    p = res["probing"]
    beaten = p["intents_beating_every_heuristic"]
    lowlat = p["per_intent"]["delay_averse"]
    n_seeds = res["n_seeds"]
    claims.append(
        _claim(
            "Learned selector beats every heuristic on 3 of 4 intents",
            beaten >= 3,
            f"beats every heuristic in all {n_seeds} seeds on {beaten}/4 intents ("
            + ", ".join(
                v["label"] for v in p["per_intent"].values() if v["wins_all_seeds"]
            )
            + ")",
        )
    )
    # The thesis states a shortfall, so a *negative* shortfall (the learned
    # selector ahead) also satisfies the claim -- but is worth naming as such.
    margin = lowlat["margin_learned_minus_best_heuristic"]
    gap, gap_hi = -margin["mean"], -margin["ci95_lo"]
    claims.append(
        _claim(
            "Within 0.0015 of lowest-latency on Low-Latency",
            bool(gap_hi <= 0.0015),
            (
                f"ahead of {lowlat['best_heuristic_label']} by {-gap:.4f}"
                if gap < 0
                else f"behind {lowlat['best_heuristic_label']} by {gap:.4f}"
            )
            + f"; worst-case shortfall across the 95% CI is {gap_hi:.4f}",
        )
    )

    c = res["ceiling"]["methods"][SHIPPED_AGENT]
    drop, light, heavy = (
        c["goodput_drop_pct"],
        c["goodput_light_mbps"],
        c["goodput_heavy_mbps"],
    )
    # "~24%" is an approximate claim; test it as "about a quarter" (the whole
    # interval in [20, 30]%) and report separately whether the two literal
    # endpoints the chapter quotes still round to 9.1 and 6.9 Gbit/s.
    endpoints_hold = (
        round(light["mean"] / 1000, 1) == 9.1 and round(heavy["mean"] / 1000, 1) == 6.9
    )
    claims.append(
        _claim(
            "Ceiling descends ~24% (9.1 -> 6.9 Gbit/s)",
            bool(20.0 <= drop["ci95_lo"] and drop["ci95_hi"] <= 30.0),
            f"{light['mean']/1000:.2f} -> {heavy['mean']/1000:.2f} Gbit/s, drop "
            f"{_fmt(drop, 2)}%; the quoted 9.1 -> 6.9 endpoints "
            + ("reproduce" if endpoints_hold else "do NOT reproduce to one decimal"),
        )
    )
    rd = c["reward_drop"]
    claims.append(
        _claim(
            "Reward stays flat across congestion (0.898 -> 0.889)",
            bool(abs(rd["mean"]) < 0.02),
            f"{_fmt(c['reward_light'], 4)} -> {_fmt(c['reward_heavy'], 4)}, "
            f"drop {_fmt(rd, 4)}",
        )
    )
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    dirs = seed_dirs(run_path, args.seeds)
    if not dirs:
        raise SystemExit(
            f"No staged seed results under {run_path/'seeds'}. "
            f"Run run_seed_result_sweep.py first."
        )
    out_dir = args.out_dir or (run_path / "seeds" / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {len(dirs)} seeds: {', '.join(d.name for d in dirs)}\n")
    res: Dict[str, Any] = {
        "run_dir": str(run_path),
        "agent": SHIPPED_AGENT,
        "seeds": [d.name for d in dirs],
        "n_seeds": len(dirs),
    }
    res["intent_matrix"] = study_intent_matrix(dirs, out_dir)
    res["intent_metrics"] = study_intent_metrics(dirs, out_dir)
    res["zeroshot"] = study_zeroshot(dirs, out_dir)
    res["pathcount"] = study_pathcount(dirs, out_dir)
    res["probing"] = study_probing(dirs, out_dir)
    res["ceiling"] = study_ceiling(dirs, out_dir)
    res["claims"] = build_claims(res)

    with open(out_dir / "claims.json", "w") as f:
        json.dump(res, f, indent=2)

    # ------------------------------------------------------------------ printout
    print("=== 1. Intent alignment: does the diagonal win every column? ===")
    for name, c in res["intent_matrix"]["columns"].items():
        mark = "OK " if c["wins_all_seeds"] else "NO "
        sig = "significant" if c["margin_ci_excludes_zero"] else "NOT significant"
        print(
            f"  {mark}{c['label']:<12} wins {c['diagonal_wins_in_seeds']}/{c['n_seeds']} seeds  "
            f"margin {_fmt(c['margin_over_best_off_diagonal'])} ({sig})  "
            f"col range {_fmt(c['column_range'], 3)}"
        )

    print("\n=== 2. Chosen-path metrics by intent ===")
    for intent, m in res["intent_metrics"]["per_intent"].items():
        print(
            f"  {INTENT_LABELS[intent]:<12} lat {m['latency_mean_ms']['mean']:6.2f} ms  "
            f"goodput {m['goodput_mean_mbps']['mean']:7.1f} Mbps  "
            f"trust {m['trust_mean']['mean']:.4f}  "
            f"loss-exposure {m['loss_exposure_pct']['mean']:.2f}%"
        )
    im = res["intent_metrics"]
    print(
        f"  Low-Latency saves {_fmt(im['latency_saving_lowlat_vs_thpt_ms'], 2)} ms "
        f"against Throughput; trust(Low-Latency) - trust(Low-Loss) = "
        f"{_fmt(im['trust_lowlat_minus_lowloss'], 4)}"
    )

    print("\n=== 3. Zero-shot interpolation ===")
    for method, z in res["zeroshot"]["methods"].items():
        print(
            f"  {z['label']:<20} latency {z['latency_at_t0_ms']['mean']:.2f} -> "
            f"{z['latency_at_t1_ms']['mean']:.2f} ms  span {_fmt(z['latency_span_ms'], 2)}  "
            f"rho in [{z['spearman_rho']['min']:.4f}, {z['spearman_rho']['max']:.4f}]  "
            f"largest step {z['largest_step_fraction']['max']:.1%} (worst seed)"
        )

    print("\n=== 4. Order invariance ===")
    for m, v in res["pathcount"]["order_invariance"].items():
        print(
            f"  {v['label']:<20} {v['total_trials']} trials  "
            f"min agreement {v['min_agreement']:.6%}  "
            f"mismatches/seed {v['mismatches_per_seed']}"
        )

    print("\n=== 5. Probing: learned selector vs heuristics, per intent ===")
    prob = res["probing"]
    learned = prob["by_method"][SHIPPED_AGENT]
    widest = prob["by_method"]["widest_path"]
    print(
        f"  goodput {learned['goodput_mbps']['mean']:.1f} Mbps "
        f"[{learned['goodput_mbps']['ci95_lo']:.1f}, {learned['goodput_mbps']['ci95_hi']:.1f}] "
        f"vs Widest-Path {widest['goodput_mbps']['mean']:.1f} "
        f"({_fmt(prob['goodput_vs_widest_pct'], 2)}%)"
    )
    print(
        f"  probe cost {learned['probe_cost_ms']['mean']:.1f} ms/sel "
        f"[{learned['probe_cost_ms']['ci95_lo']:.1f}, {learned['probe_cost_ms']['ci95_hi']:.1f}] "
        f"vs Widest-Path {widest['probe_cost_ms']['mean']:.0f} ms "
        f"({_fmt(prob['probe_cost_reduction_vs_widest'], 1)}x cheaper)"
    )
    for profile, v in res["probing"]["per_intent"].items():
        mark = "OK " if v["wins_all_seeds"] else "NO "
        print(
            f"  {mark}{v['label']:<12} learned {_fmt(v['learned_reward'])} vs "
            f"{v['best_heuristic_label']} {_fmt(v['best_heuristic_reward'])}  "
            f"margin {_fmt(v['margin_learned_minus_best_heuristic'])}"
        )

    print("\n=== 6. Single-path ceiling ===")
    for m, v in res["ceiling"]["methods"].items():
        print(
            f"  {v['label']:<20} goodput {v['goodput_light_mbps']['mean']/1000:.2f} -> "
            f"{v['goodput_heavy_mbps']['mean']/1000:.2f} Gbit/s "
            f"({_fmt(v['goodput_drop_pct'], 1)}%)   reward "
            f"{v['reward_light']['mean']:.3f} -> {v['reward_heavy']['mean']:.3f}"
        )

    print("\n=== Claim survival ===")
    for c in res["claims"]:
        print(f"  [{'SURVIVES' if c['survives'] else 'FAILS   '}] {c['claim']}")
        print(f"             {c['detail']}")

    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
