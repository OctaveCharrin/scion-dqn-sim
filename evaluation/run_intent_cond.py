#!/usr/bin/env python3
"""Intent conditioning orchestrator: run all intent-conditioning evaluations, write CSVs +
LaTeX table, and render figures into one timestamped artifact directory.

Usage:
    uv run python run_intent_cond.py [run_dir] [--metric goodput|reward] [--max-pairs N]

Creates ``<run_dir>/intent_cond_<YYYYMMDD_HHMMSS>/`` containing:
    ablation_reward_matrix.csv, ablation_behavioral_divergence.csv, table_6_1.tex
    intent_reward_matrix.csv, intent_selection_metrics.csv
    probing_quality.csv, ceiling_by_congestion.csv
    figures/fig_6_1_heatmap.png, fig_6_1_boxplots.png,
            fig_6_2_quality_vs_probe.png, fig_6_3_ceiling.png

Prerequisite: the run dir must already hold the trained checkpoints
(``dqn_model.pth``, ``dqn_conditional_concat_model.pth``,
``dqn_conditional_scoring_model.pth``). See the step-04 training scripts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.pipeline.intent_cond_eval import (
    MAX_EVAL_PAIRS,
    run_ablation,
    run_intent_alignment,
    run_probing_ceiling,
)
from src.pipeline.intent_cond_figures import generate_all_figures
from src.pipeline.run_dirs import resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument(
        "--metric",
        choices=["goodput", "reward"],
        default="goodput",
        help="QoE metric for Figs 6.2/6.3 (default: goodput).",
    )
    parser.add_argument("--max-pairs", type=int, default=MAX_EVAL_PAIRS)
    parser.add_argument("--congestion-bins", type=int, default=6)
    args = parser.parse_args()

    t0 = time.time()
    run_path = Path(args.run_dir or resolve_run_dir())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = run_path / f"intent_cond_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Convenience alias so the figures CLI can find the newest set.
    print(f"\nIntent conditioning pipeline on {run_path}")
    print(f"  artifact dir: {out_dir}")

    summary = {
        "run_dir": str(run_path),
        "artifact_dir": str(out_dir),
        "metric": args.metric,
        "max_pairs": args.max_pairs,
    }

    print("\n[1/4] Ablation (Table 6.1) ...")
    abl = run_ablation(run_path, out_dir, max_pairs=args.max_pairs, progress=print)
    summary["ablation"] = {
        "methods": abl["methods"],
        "divergence": {
            r["method"]: r["behavioral_divergence"] for r in abl["divergence_rows"]
        },
    }

    print("\n[2/4] Intent alignment (Figure 6.1) ...")
    align = run_intent_alignment(
        run_path, out_dir, max_pairs=args.max_pairs, progress=print
    )
    summary["intent_alignment"] = {"matrix_csv": align["matrix_csv"]}

    print("\n[3/4] Probing overhead + ceiling (Figures 6.2/6.3) ...")
    pc = run_probing_ceiling(
        run_path,
        out_dir,
        max_pairs=args.max_pairs,
        n_congestion_bins=args.congestion_bins,
        progress=print,
    )
    summary["probing_ceiling"] = {
        "quality_csv": pc["quality_csv"],
        "ceiling_csv": pc["ceiling_csv"],
    }

    print("\n[4/4] Rendering figures ...")
    figs = generate_all_figures(out_dir, metric=args.metric)
    summary["figures"] = figs
    for name, path in figs.items():
        print(f"  {name}: {path}")

    with open(out_dir / "intent_cond_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts in {out_dir}")
    print(f"  LaTeX table: {out_dir / 'table_6_1.tex'}")


if __name__ == "__main__":
    main()
