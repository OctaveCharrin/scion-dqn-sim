#!/usr/bin/env bash
# Seed variance for the conditioning ablation (Chapter 4, sec:p1eval:ablation).
#
# Every number in the chapter comes from a single training seed, and the reported
# spread is across selections rather than across runs. This retrains the three
# conditional variants under N seeds and re-runs the ablation on each, so the
# load-bearing comparisons (FiLM 0.732 vs Value-Concat 0.708 on Low-Latency;
# FiLM 0.128 vs Two-Stream 0.149 adaptivity) can carry confidence intervals.
#
# Only torch/numpy/random are reseeded: the environment's pair, hour and profile
# RNGs stay fixed, so every seed sees the identical stream of training contexts
# and only the learning process varies.
#
# Usage: ./run_seed_sweep.sh <run_dir> [seeds...]
set -euo pipefail

RUN="${1:?usage: run_seed_sweep.sh <run_dir> [seeds...]}"
shift || true
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(1 2 3 4 5); fi

for S in "${SEEDS[@]}"; do
  DIR="$RUN/seeds/seed$S"
  mkdir -p "$DIR"
  # Shadow run dir: the run context is shared, only the checkpoints differ.
  for artifact in topology selected_pair.json path_store.json link_states.pkl; do
    [ -e "$DIR/$artifact" ] || ln -s "$(realpath "$RUN/$artifact")" "$DIR/$artifact"
  done

  echo "=== seed $S: FiLM ==="
  uv run python 04_train_conditional_dqn.py "$RUN" --seed "$S" \
    --checkpoint "$DIR/dqn_conditional_scoring_model.pth" \
    --stats-json "$DIR/dqn_conditional_training_stats.json"

  echo "=== seed $S: value-concat ==="
  uv run python 04_train_conditional_value_concat_dqn.py "$RUN" --seed "$S" \
    --checkpoint "$DIR/dqn_conditional_value_concat_model.pth" \
    --stats-json "$DIR/dqn_conditional_value_concat_training_stats.json"

  echo "=== seed $S: two-stream concat ==="
  uv run python 04_train_conditional_concat_dqn.py "$RUN" --seed "$S" \
    --checkpoint "$DIR/dqn_conditional_concat_model.pth" \
    --stats-json "$DIR/dqn_conditional_concat_training_stats.json"

  echo "=== seed $S: ablation ==="
  uv run python eval_ablation_intent.py "$DIR" --out-dir "$DIR/ablation" --max-pairs 32
done

echo "seed sweep done: $RUN/seeds"
