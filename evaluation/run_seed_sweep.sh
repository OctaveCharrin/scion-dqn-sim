#!/usr/bin/env bash
# Seed variance study.
#
# Every number in the study would otherwise come from a single training seed,
# with the reported spread taken across selections rather than across runs. This
# retrains all four rungs of the ablation ladder under N seeds and re-runs the
# ablation on each, so every row of the table can carry a confidence interval --
# not just the conditional variants, whose gaps are the smallest but whose rows
# are not the only ones the study leans on (the flat-vs-scoring goodput gap is
# the study's first summary claim).
#
# The four rungs are the ones the thesis reports: Flat DQN, the unconditioned
# Scoring DQN, Value-Concat, and Two-Stream-Concat. FiLM was dropped from the
# thesis on 2026-07-26 -- it tied within seed noise at more parameters -- and is
# deliberately not trained here, so no reported number can pick it up.
#
# Only torch/numpy/random are reseeded: the environment's pair, hour and profile
# RNGs stay fixed, so every seed sees the identical stream of training contexts
# and only the learning process varies.
#
# Idempotent: an agent whose checkpoint already exists in the seed dir is skipped,
# so the script can be re-run to fill in newly added agents without redoing the
# trainings that are already on disk. Delete a checkpoint to force a retrain.
#
# The ablation is the only study run here. For the study's other five results
# (intent alignment, zero-shot interpolation, path-count scaling and order
# invariance, probing overhead, and the congestion ceiling), run
# ``run_seed_result_sweep.py`` and then ``analyze_seed_results.py`` against the
# same seed directories.
#
# Usage: ./run_seed_sweep.sh <run_dir> [seeds...]
set -euo pipefail

# train <label> <script> <checkpoint> <stats-json> <seed>
train() {
  local label="$1" script="$2" ckpt="$3" stats="$4" seed="$5"
  if [ -e "$ckpt" ]; then
    echo "=== seed $seed: $label (already trained, skipping) ==="
    return
  fi
  echo "=== seed $seed: $label ==="
  uv run python "$script" "$RUN" --seed "$seed" --checkpoint "$ckpt" --stats-json "$stats"
}

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

  train "flat DQN" 04_train_dqn.py \
    "$DIR/dqn_model.pth" "$DIR/training_stats.json" "$S"

  train "scoring DQN (uncond.)" 04_train_scoring_enhanced_dqn.py \
    "$DIR/dqn_scoring_enhanced_model.pth" \
    "$DIR/dqn_scoring_enhanced_training_stats.json" "$S"

  train "value-concat" 04_train_conditional_value_concat_dqn.py \
    "$DIR/dqn_conditional_value_concat_model.pth" \
    "$DIR/dqn_conditional_value_concat_training_stats.json" "$S"

  train "two-stream concat" 04_train_conditional_concat_dqn.py \
    "$DIR/dqn_conditional_concat_model.pth" \
    "$DIR/dqn_conditional_concat_training_stats.json" "$S"

  echo "=== seed $S: ablation ==="
  uv run python eval_ablation_intent.py "$DIR" --out-dir "$DIR/ablation" --max-pairs 32
done

echo "seed sweep done: $RUN/seeds"
