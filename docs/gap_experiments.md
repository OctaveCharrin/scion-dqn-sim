# Closing the Chapter 4 gaps — a work order for this repo

**Audience:** an engineer or coding agent working in `/home/octav/fork-scion-dqn-sim`.
**Written:** 2026-07-25. Every CLI signature below was verified against the source on that date.
**Companion to:** the 21 `\gap{}` markers in `/home/octav/thesis-report/sections/4_single_path.tex`, and
the older, broader figure catalogue in `thesis_improvements.tex` (same directory).

---

## 1. Context — what this repo is for, and what is wrong with it

This repo is the simulator and RL codebase behind **Chapter 4 of a master's thesis** ("Motivating Study:
Intent-Conditioned Single-Path Selection", ETH Zurich, submission 2026-08-25). Chapter 4 makes four
claims, and every number and figure in it comes from the single run
`evaluation/run_20260722_180329/chapter6_20260722_200655/`:

1. **Intent conditioning.** One policy, conditioned on a 5-D intent vector `w = (w1,w2,w3,w4,w_probe)`,
   serves many objectives without retraining — where the conditioning is applied via **FiLM** modulation
   of per-path features.
2. **Permutation-equivariant scoring.** A shared per-path scorer handles a variable candidate-path count
   `N`, replacing a fixed-action DQN.
3. **Probing reduction.** The learned selector matches widest-path goodput at ~40× less probing.
4. **The single-path ceiling.** Even optimal single-path selection loses ~25 % goodput under congestion,
   which motivates the multipath work in Chapter 5.

A 2026-07-25 audit compared the chapter against this code and found that **claims 1 and 2 are not
actually supported by the experiments that were run**, and that a substantial amount of measured data was
never reported. A `\gap{}` in the thesis is a visible red TODO marking exactly one such hole. This
document is the execution plan for closing them.

### 1.1 The one finding that matters most

`chapter6_summary.json` records a **fourth** conditioning architecture that the chapter never mentions:

| method | checkpoint | adaptivity |
|---|---|---|
| `flat_dqn` | `dqn_model.pth` | 0.011 |
| `conditional_concat` (value-stream only) | `dqn_conditional_value_concat_model.pth` | 0.050 |
| **`conditional_concat_2stream`** (intent in **both** streams) | `dqn_conditional_concat_model.pth` | **0.149** |
| `conditional_film` | `dqn_conditional_scoring_model.pth` | 0.128 |

It was excluded at `src/pipeline/chapter6_eval.py:69-78` with the comment *"statistically tied with FiLM
on quality but adds no principle, so the chapter reports the clean three-way."*

Why this is a problem: the reported `conditional_concat` puts the intent in the **value stream only**. In
a dueling head the value stream is path-independent, so `Q_i = V(s,w) + (A_i − mean A)` and the intent
cancels in every pairwise comparison — that variant **cannot re-rank by construction**, which the repo
itself proves with a unit test (`tests/test_chapter6_eval.py:193`), not an experiment. So the chapter
reported a 2.5× adaptivity win over a baseline that is architecturally incapable of competing, while the
baseline that *is* capable, and that beats FiLM, sits unreported in the same JSON.

The thesis has since been rewritten around the narrower, provable statement (Proposition 4.1: conditioning
entering only path-independent terms cannot change an `argmax`). **Task 2.1 below is the highest-priority
item in this document**, because it decides how the chapter's central argument is framed.

---

## 2. Standing rules for anyone executing this work

These are not stylistic preferences; violating them either destroys published artifacts or produces
numbers that cannot be compared to the ones in the thesis.

1. **Never write into `run_20260722_180329/chapter6_20260722_200655/`.** Those CSVs and PNGs are what the
   current thesis draft cites. Always pass `--out-dir` / `--artifact-dir` pointing at a *new* directory.
2. **Always pass the run directory explicitly.** Every script defaults to `resolve_run_dir()` = newest
   `run_*`. The moment task 3.6 generates a second topology, every unqualified command silently retargets.
3. **Do not delete or overwrite `.pth` checkpoints.** `dqn_conditional_concat_model.pth` in particular is
   irreplaceable without a retrain, and its reward matrix was already lost once to an overwrite.
4. **Do not change seeds** except in task 2.5, whose entire point is seed variance. Seeds are currently
   hard-coded (`rng_seed=42` train / `7` eval, `pair_rng_seed=123`, `hour_rng_seed=456`,
   `weight_rng_seed=789`, topology/traffic `42`).
5. **Preserve existing CSV column names** when adding columns. `src/pipeline/chapter6_figures.py` and the
   thesis figures read them positionally in places.
6. **Run `uv run python -m pytest` from the repo root after any code edit.** Several of these tasks touch
   `chapter6_eval.py`, which is covered by tests that encode the conditioning claims.
7. **Report the actual numbers back, including ones that contradict the thesis.** A result that undercuts
   a claim is the *purpose* of this exercise, not a failure to be smoothed over. Task 2.1 is expected to
   produce one.
8. **Do not edit anything in `/home/octav/thesis-report`.** Report findings; the thesis author decides the
   wording.

### 2.1 Preconditions

```bash
cd /home/octav/fork-scion-dqn-sim/evaluation
export RUN=run_20260722_180329          # the only run on disk

uv run python -m pytest                 # from repo root — must pass before you start
ls $RUN/*.pth                           # expect exactly 4 checkpoints
```

Note: `eval_conditional_dqn_rewards.py` has **no argparse at all** — it only ever uses the newest run
dir. Adding a `run_dir` positional is a prerequisite for using it safely (task 3.2).

---

## 3. Priority summary

| # | Task | Why it matters | Effort | Priority |
|---|---|---|---|---|
| 5.1 | Two-stream concat row | Decides the framing of the chapter's central claim | 2-line edit + 10 min run | **P0** |
| 4.1 | Reward-vs-congestion panel | Makes the Ch4→Ch5 bridge argument properly | 1 command | **P0** |
| 6.1 | Zero-shot intent sweep | Claim 1 is currently untested | New script | **P0** |
| 4.4 | Train unconditioned scoring agent | Claim 1 has no architectural control | ~1 h train | **P1** |
| 5.4 | Genuinely held-out pairs | Current split is temporal only | Small edit | **P1** |
| 4.2 | Step 05/06 — inference latency | §6.2.1 latency gap; no data exists | 3 commands | **P1** |
| 5.5 | Seed variance | The load-bearing 0.732-vs-0.708 gap has no error bar | 5× train cost | **P1** |
| 5.2 | Oracle upper bound | "near-oracle" is currently unsupported | Small edit | **P2** |
| 6.2 | Per-`N` breakdown | Claim 2 is unmeasured | Small edit + caveat | **P2** |
| 4.3, 4.5–4.7, 5.3, 6.3–6.7 | Supporting material | Fills specific `\gap{}`s | Varies | **P2–P3** |

---

## 4. Runnable today — no code changes

### 4.1 Reward-vs-congestion ceiling panel — P0

**Why.** Chapter 4 ends by arguing that the single-path *ceiling* is architectural, not a policy failure,
and that this is what motivates multipath. The strongest evidence for that is already measured and never
plotted: FiLM's **reward** is flat across congestion (0.896 → 0.886) while its **goodput** falls 24.5 %
(9067 → 6844 Mbps). A two-panel figure showing "the policy keeps doing its job while the architecture
runs out of headroom" makes the bridge to Chapter 5 far better than prose. The `reward_mean` column is
already in `ceiling_by_congestion.csv`.

```bash
cp -r $RUN/chapter6_20260722_200655 $RUN/chapter6_reward_metric
uv run python 06b_generate_chapter6_figures.py $RUN \
  --artifact-dir $RUN/chapter6_reward_metric --metric reward
```

⚠️ Without the `cp` this **overwrites** the published `fig_6_2_*.png` / `fig_6_3_*.png`. See rule 1.

**Done when:** `chapter6_reward_metric/fig_6_3_ceiling.png` shows reward on the y-axis, with FiLM and
Widest-Path near-flat and Shortest-Path / Random collapsing (0.72→0.31 and 0.26→−0.32).

### 4.2 Inference latency and Figures 1–4 — P1

**Why.** The thesis claims the controllers are deployable and §6.2.1 carries a `\gap{}` asking for
measured per-decision inference latency. **No latency data exists anywhere in the repo.** It is recorded
only by `05_evaluate_methods.py` as `avg_selection_time_ms`, and that script has never been run on this
run directory. The same run also produces the per-profile reward heatmap and the probe breakdown.

```bash
uv run python 05_evaluate_methods.py $RUN          # → evaluation_results.json (incl. selection time)
uv run python eval_multi_reward_comparison.py $RUN # → multi_reward_comparison.json
uv run python 06_generate_figures.py $RUN          # → figure1..figure4
```

⚠️ Do task 5.3 (`.eval()` parity) **first** — otherwise dropout is active during step-05 inference and
the numbers are not comparable to the Chapter 6 harness.

**Done when:** `evaluation_results.json` contains `avg_selection_time_ms` per method, and you can state
whether the conditional agent's per-decision inference fits inside a plausible selection budget.

### 4.3 `w_probe` sensitivity — P2

**Why.** Chapter 4 excludes the two probe-frugality profiles (`probe_minimal` w_probe=0.001,
`probe_averse` w_probe=0.35) from all conditioning studies, on the stated grounds that `w_probe` is
path-independent. **That justification is wrong:** probe cost is charged per hop (latency `10 + 0.5·hops`,
full `100 + 20·hops`), so a probe-frugal intent does have an incentive to prefer short paths. Meanwhile
those two profiles consume **a third of the training schedule**. `eval_multi_reward_comparison.py` (run in
4.2) is the only place their effect is ever measured.

```bash
uv run python -c "
import json; d=json.load(open('$RUN/multi_reward_comparison.json'))
print(json.dumps(d, indent=2)[:4000])"
```

**Done when:** you can report, for `low_probe_cost` vs `high_probe_cost`, whether mean probe cost per
selection and mean chosen hop count actually differ. Either result is publishable — it either justifies
the exclusion quantitatively or shows the exclusion was a mistake.

### 4.4 Train the missing unconditioned agents — P1

**Why.** `04_train_all_models.py` defines five variants, but this run only ever produced four
checkpoints, and **all three unconditioned path-scoring agents are missing**. The enhanced path-scoring
DQN is the exact architectural control for the chapter's core question: the conditional agents beat the
flat DQN by ~44 % on goodput (8.2 vs 5.65 Gbps), and without this control that gap cannot be attributed
to *conditioning* rather than to *per-path scoring*. As written, Chapter 4 attributes it to conditioning.

```bash
uv run python 04_train_scoring_enhanced_dqn.py $RUN   # → dqn_scoring_enhanced_model.pth  (the control)
uv run python 04_train_scoring_dqn.py         $RUN    # → dqn_scoring_simple_model.pth
uv run python 04_train_simple_dqn.py          $RUN    # → dqn_simple_model.pth
```

Each is ~533 episodes × 24 steps. Re-run 4.2 afterwards to pick them up.

**Done when:** the enhanced path-scoring agent's goodput and per-intent reward are known, so the
conditioning gap can be reported net of the scoring-architecture gap.

### 4.5 Cross-profile action diversity — P3

**Why.** An independent check on the Adaptivity metric, which is doing a lot of work in Table 4.1 and is
defined in a slightly unusual way (`(distinct − 1) / (min(4, N) − 1)`). This script measures action
agreement and cross-profile diversity by a different route; agreement between the two strengthens the
claim considerably.

```bash
uv run python analyze_conditional_sensitivity.py $RUN \
  --max-pairs 32 --out $RUN/conditional_sensitivity.json
```

Default `--max-pairs` is 16 over only 24 hours; raise it to match the 32 pairs × 336 hours used elsewhere.

### 4.6 Environment-realism numbers — P2

**Why.** §4.1.1 asserts that the prior work's environment lacked the fidelity to make path selection
matter, and §4.1.2 asserts this one has it. **Neither side is measured.** These JSONs supply the
supporting table.

```bash
uv run python inspect_traffic.py $RUN    # refreshes traffic_inspection.json
```

Already on disk: `simulation_metadata.json` (p90 utilization 0.76, **7.8 % of links oversubscribed**),
`traffic_inspection.json` (**6.8 % of pair-hours with no usable bandwidth**), `beaconing_stats.json`
(29.95 candidate paths/pair over 8010 routable pairs, 1004 core + 1127 up/down segments).

**Ideally also compute** the statistic the thesis actually needs and nobody has: *the spread of achievable
goodput across a pair's candidate paths within a decision context*. If that spread is wide and varies by
hour, path choice demonstrably matters; if it is narrow, every effect in Chapter 4 is small. This is the
single most useful number in this section.

### 4.7 Topology figures — P3

**Why.** Three publication-quality figures showing the BRITE → SCION → peering conversion already exist
and are unused. Chapter 4 currently describes that conversion in prose only, and the conversion matters
more than the reader would guess — BRITE is run at 50 nodes with *all* links at exactly 10 Gbps, and the
90 ASes / 528 links with heterogeneous capacity come from synthetic structure the converter adds.

```bash
cp $RUN/topology/step2_scion_enhanced.png /home/octav/thesis-report/figures/p1env_topology.png
```

---

## 5. One small code edit, then run

### 5.1 The two-stream concat row — P0, do this first

**Why.** See §1.1. This is the difference between an honest chapter and one an examiner can dismantle.

**Edit** `src/pipeline/chapter6_eval.py`:

```python
CONDITIONAL_CHECKPOINTS: Dict[str, str] = {
    "conditional_film":           "dqn_conditional_scoring_model.pth",
    "conditional_concat":         "dqn_conditional_value_concat_model.pth",
    "conditional_concat_2stream": "dqn_conditional_concat_model.pth",   # ADD
}
METHOD_LABELS["conditional_concat_2stream"] = "Two-Stream-Concat"       # ADD
```

Also update the stale comment at lines 69-78, which currently documents the exclusion. The checkpoint
already exists — **no retraining is needed**; only the reward matrix was lost to an overwrite.

```bash
uv run python eval_ablation_intent.py $RUN --out-dir $RUN/chapter6_4way --max-pairs 32
```

**Done when:** `chapter6_4way/ablation_reward_matrix.csv` and `table_6_1.tex` contain four methods.
**Report back explicitly:** the four per-intent rewards for `conditional_concat_2stream`, and whether it
ties, beats, or loses to FiLM on **Low-Latency** specifically — that is the intent that requires
re-ranking and the one the chapter's argument turns on.

**Expected outcome and what it implies.** Adaptivity 0.149 > FiLM's 0.128 is already known. If the rewards
are also tied, the defensible thesis claim becomes *"FiLM matches per-path concatenation while
guaranteeing re-ranking structurally, at +5.2 % parameters"* rather than *"modulation is required"*. That
is still a real contribution; it is just a different one.

### 5.2 Oracle upper bound — P2

**Why.** Chapter 4 twice claimed to reproduce a *near-oracle* result, but **no oracle is evaluated**.
Widest-path is a heuristic that maximizes bandwidth — the right target only under the Throughput intent.
A true oracle (per-context argmax of the actual reward under the intent being scored) would give a
per-intent upper bound and turn Table 4.1 into a normalized optimality gap. The word "near-oracle" has
been removed from the thesis pending this.

An oracle exists — `_eval_oracle_greedy` in `evaluation/eval_conditional_dqn_rewards.py:82` — but has
never been run, **and it is not currently comparable**: it passes `max_possible_bw=goodput_cap`
(10 000 Mbps) while the agents' reward normalizes by the per-hour, per-pair best candidate path.

**Edit:** at `eval_conditional_dqn_rewards.py:471`, drop the `goodput_cap` argument so `compute_reward`
falls back to `_max_path_bandwidth_at_current_hour()`. Add a `run_dir` positional while you are there.

```bash
uv run python eval_conditional_dqn_rewards.py $RUN   # → conditional_dqn_reward_eval.json
```

**Done when:** oracle and agent rewards are on the same normalization, and the FiLM optimality gap can be
stated per intent.

### 5.3 Evaluation-mode parity — P1, blocks 4.2

**Why.** `chapter6_eval.py:143-147` sets `epsilon = 0` **and** calls `q_network.eval()`, with an explicit
comment that otherwise the measured divergence would be dropout noise. `05_evaluate_methods.py:96-135`
sets `epsilon = 0` but **never calls `.eval()`**, so dropout 0.1 is live. The two harnesses therefore
measure different things, and any latency or reward number from step 05 is not comparable to Chapter 4's.

**Edit:** add `.eval()` to the agent loading in `05_evaluate_methods.py`, then re-run 4.2.

### 5.4 Genuinely held-out pairs — P1

**Why.** Chapter 4 said "held-out pairs" three times. It is false: `chapter6_eval.py` evaluates
`pair_pool[:32]`, which are **all from the same 8010-pair pool used in training** (`DQN_TRAIN_PAIR_CAP=64`
means the first 64 pairs are trained on) and **all share source AS 0**. The split is temporal only. This
leaves the *Structural & Topological Generalization* property of the thesis's problem statement entirely
unevaluated, and it is cheap to fix.

**Edit:** add a `--pair-offset` flag (or a stride) to `chapter6_eval.py`'s pair selection so evaluation can
draw from `pair_pool[64:]`, ideally sampling across several source ASes rather than a contiguous slice.

```bash
uv run python eval_ablation_intent.py $RUN --out-dir $RUN/chapter6_heldout \
  --max-pairs 32 --pair-offset 64
uv run python eval_probing_ceiling.py $RUN --out-dir $RUN/chapter6_heldout \
  --max-pairs 32 --pair-offset 64
```

**Done when:** Table 4.1 can be reported on pairs the agent never trained on, and the degradation (if any)
versus the in-pool numbers is quantified.

### 5.5 Seed variance — P1

**Why.** **Every number in Chapter 4 comes from a single training seed.** The reported `std` is across
selections within one run, not across runs, so it says nothing about reproducibility. The chapter's
load-bearing comparison is a **0.024 reward gap** (FiLM 0.732 vs Value-Concat 0.708 on Low-Latency), and
the FiLM-vs-two-stream adaptivity comparison from 5.1 is 0.128 vs 0.149 — both well inside the range a
seed change could plausibly move.

Seeds are hard-coded. `ScoringHyperparams.from_env` (`src/rl/path_selection_train.py:98-116`) exposes ten
knobs but no seed.

**Edit:** add `"DQN_SEED": ("seed", int)` to that mapping and thread it through `train_flat_dqn`,
`train_scoring_dqn`, and `train_conditional_scoring_dqn` into torch/numpy/random seeding and the env
constructors.

```bash
for S in 1 2 3 4 5; do
  DQN_SEED=$S uv run python 04_train_conditional_dqn.py $RUN \
    --checkpoint $RUN/film_seed$S.pth --stats-json $RUN/film_seed${S}_stats.json
  DQN_SEED=$S uv run python 04_train_conditional_value_concat_dqn.py $RUN \
    --checkpoint $RUN/vconcat_seed$S.pth --stats-json $RUN/vconcat_seed${S}_stats.json
done
```

~5× current training cost per architecture. Pair with task 6.4 for the significance test.

**Done when:** Table 4.1 can carry means with 95 % confidence intervals over ≥5 seeds.

---

## 6. Needs new code

### 6.1 Zero-shot intent generalization — P0, the biggest hole

**Why.** The abstract, Contribution 1, §4.2.3, and §6.2.3 all claim the policy generalizes **zero-shot**
across intents, including interpolations between trained profiles. **Every intent ever evaluated is one of
the six profiles the agent trained on.** The results therefore establish only that one network can store
six behaviors — not that it has learned a mapping over the intent space. No code exists for this at all.

**Write** `evaluation/eval_intent_interpolation.py`:

- Sweep `w(t) = (1−t)·bandwidth_max + t·delay_averse` for `t ∈ {0, 0.1, …, 1.0}` (these two genuinely
  conflict; `loss_averse` and `balanced_extreme` do not — see §7).
- For each `t`: set `env.reward_weights`, call `observe_scoring_conditional()`, select, and log chosen
  latency, goodput, trust, and the reward under both endpoint objectives.
- Same 32 pairs × 336 held-out hours as everywhere else; `epsilon=0` and `.eval()`.
- Run it for **both** FiLM and Value-Concat, since a conditioning mechanism that fails to interpolate is
  worth knowing about regardless of which one ships.

```bash
uv run python eval_intent_interpolation.py $RUN --steps 11 --out-dir $RUN/zeroshot
```

**Done when:** you can say whether chosen latency moves *monotonically* from ~17 ms to ~12.4 ms as `t`
goes 0→1. A smooth curve supports the zero-shot claim; a step function at `t≈0.5` shows the network
memorized six points, which would mean the claim must be dropped from the abstract.

**Companion experiment (extrapolation):** retrain with `DQN_CONDITIONAL_PROFILES` restricted to five
profiles and evaluate on the held-out sixth, comparing against an agent trained on that profile alone.

### 6.2 Per-`N` breakdown and order invariance — P2

**Why.** The permutation-equivariant scoring architecture is Contribution 2 and the stated reason for
abandoning the fixed-action DQN, but **no result is broken down by candidate-path count**, so the claim
rests entirely on the architecture's construction.

**Important caveat — read before starting.** In this topology **7988 of 8010 pairs have exactly 30
paths** (distribution: `{30: 7988, 27: 8, 28: 2, 1: 8, 4: 2, 3: 2}`). There is almost no variation in `N`
to measure, so a straight binning will be uninformative. Two viable approaches:

- **Subsample at decision time:** randomly restrict the candidate set to `N ∈ {2, 4, 8, 16, 30}` before
  scoring, and report quality per `N`. This tests the actual claim and is cheap.
- **Order invariance:** re-score each context under permuted path orderings and assert the chosen path is
  unchanged. This is a direct test of permutation equivariance and belongs in `tests/` as well.

**Edit:** add `n_paths` to the per-selection rows emitted by `run_ablation` regardless, so the breakdown
is possible later.

### 6.3 Training curves — P2

**Why.** No convergence evidence appears anywhere in the thesis, and the appendix carries a `\gap{}` for
it. The data is fully saved and unused. It also surfaces a problem: the **flat DQN's mean training reward
is −0.030 and its loss *rises*** over the run (0.130 → 0.174), so the weakest rung of the ablation ladder
may be undertrained rather than architecturally limited — which would inflate the gap Chapter 4 attributes
to conditioning.

**Write** a plotting script over the four `*_training_stats.json` files in `$RUN`. Each contains
`episode_rewards` (533 or 666 values), `losses`, `episode_probes`, and — for the conditional models —
`episode_weight_profiles`, which enables **per-intent learning curves** (~111 episodes per profile under
the stratified schedule). Those per-profile curves are the only place `probe_minimal` and `probe_averse`
appear at all.

Known final values for validation: flat −0.216 → +0.049; FiLM −0.057 → **+0.822**; two-stream concat
−0.035 → +0.814; value concat −0.051 → +0.816 (first-50 vs last-50 episode means).

### 6.4 Paired significance test — P1, pairs with 5.5

**Why.** See 5.5. With 10 752 shared decision contexts a paired test is easy and would settle the
FiLM-vs-concat question properly.

**Edit:** `run_ablation` currently aggregates to means and discards per-context rewards. Persist the raw
per-context reward arrays (they are already computed), then run a paired Wilcoxon signed-rank test between
methods on identical contexts.

### 6.5 Per-intent probing study — P2

**Why.** `run_probing_ceiling` hard-codes `balanced_extreme` — the one intent where the learned selector's
advantage is smallest and where a single heuristic (widest-path) is close to the right rule. The stronger
and more interesting claim, which the thesis discussion already asserts, is that **one conditioned policy
tracks the per-intent strongest heuristic** — widest-path under Throughput, lowest-latency under
Low-Latency — at a fraction of any of their probing costs.

**Edit:** parameterize `run_probing_ceiling` by profile and emit one row per (method, intent).

Also report **probes per selection** alongside milliseconds: FiLM issues **1** probe at 135 ms; every
heuristic except widest-path issues **30** at 359.7 ms; widest-path issues 30 at 5388 ms. "1 versus 30" is
the more legible quantity, and note that FiLM is the *cheapest* method overall — the chapter previously
had this backwards.

### 6.6 ECMP realism — P3

**Why.** `ECMPSelector` hashes `(source_as, destination_as, flow_id)`, but `chapter6_eval.py:214-220`
passes a constant `flow_id=0`, so every flow of a pair pins to the same path and ECMP exhibits **none** of
the load spreading its name implies. In `05_evaluate_methods.py:284` the flow stub lacks the keys
entirely, so the hash key is `(None, None, None)` for every pair and hour. Either vary `flow_id` to make
it a real ECMP baseline, or rename it to reflect what it does (hash-pinned shortest path).

### 6.7 Larger-topology generalization — P3

**Why.** *Structural & Topological Generalization* is a stated required property and is entirely
unevaluated — only one topology exists.

`EVAL_BRITE_N_NODES` exists, so *generating* a larger topology is free. Evaluating an **existing**
checkpoint against a **new** run dir is **not** supported and needs a cross-run loader.

```bash
EVAL_BRITE_N_NODES=150 uv run python 01_generate_topology.py
uv run python 02_run_beaconing.py    run_<new>
uv run python 03_simulate_traffic.py run_<new>
# then: new code to load $RUN's checkpoint against run_<new>'s environment
```

Two scaling traps: `beacon_pipeline.py` switches to pair sampling with a 200-pair cap above
`MAX_NODES_FULL_PAIR_SCAN = 200` nodes, and `DQN_TRAIN_PAIR_CAP = 64` means the training episode budget
does **not** grow with topology size.

---

## 7. Known-weak results worth understanding before you start

These are documented in the thesis and are *not* bugs, but they shape what any new experiment should
target.

- **Two of the four evaluated intents are not distinct.** `loss_averse` and `balanced_extreme` produce
  near-identical selections on every metric. In the reward heatmap the Low-Loss column spans 0.01 and the
  Balanced column's diagonal is actually *beaten* by the Throughput row (0.89 vs 0.90). Intent alignment
  is therefore demonstrated on **2 of 4** intents. Any new intent profile should be chosen so its optimum
  genuinely conflicts with high bandwidth — e.g. hop-count-averse or trust-dominant.
- **The trust metric and the Low-Loss profile are misaligned.** Trust is reported with canonical
  `w3 = w4 = 0.5`, so the short paths chosen by the *delay-averse* intent maximize it (0.935) while
  `loss_averse` (w3=1.0, w4=0.15) optimizes something the canonical trust score largely ignores (0.919).
  The boxplot figure's third panel is titled "Low-Loss intent → higher" and is contradicted by its own
  data. Either score trust under each intent's own `w3, w4`, or plot chosen loss rate instead.
- **The reward is relative, not absolute.** Goodput is normalized by the **best candidate path for that
  pair at that hour** (`evaluation_env.py:315-321`), not by the per-topology cap (which is computed,
  stored in every checkpoint, and never passed to `compute_reward`). This is why FiLM's reward is flat
  across congestion while goodput falls — the reward measures *selection quality* and is blind to the
  ceiling by construction.
- **The probe penalty is a per-probe mean**, not a total (`_effective_probe_cost_ms`, ÷ `num_probes`,
  reference 500 ms). So a method issuing 30 cheap probes is penalized *less* than one issuing a single
  expensive probe. Widest-path is charged 179.6 ms and FiLM 135 ms — comparable. FiLM's reward edge over
  widest-path (0.8939 vs 0.8913) comes from the **trust term**, not from probe accounting.
- **The bandwidth model is open-loop.** `available = max(0, C_e − load_e)` at the bottleneck; the agent's
  own selection never adds load. "Goodput" is spare capacity, not an achieved rate. Utilization is capped
  at 1.5, not 1.0, which is why the ceiling plot's x-axis reaches 1.4.
- **Training is short.** 533/666 episodes × 24 steps with a gradient step every 4 → roughly 3 200/4 000
  optimizer updates total. `epsilon` decays only to 0.035, never reaching its 0.01 floor. PER `beta`
  anneals over 50 000 steps, so it barely moves.

---

## 8. Suggested execution order

```bash
cd /home/octav/fork-scion-dqn-sim/evaluation && export RUN=run_20260722_180329

# --- P0, cheap: minutes, no code edits -------------------------------------
cp -r $RUN/chapter6_20260722_200655 $RUN/chapter6_reward_metric
uv run python 06b_generate_chapter6_figures.py $RUN --artifact-dir $RUN/chapter6_reward_metric --metric reward

# --- P0: one 2-line edit (5.1), then the result that reframes the chapter ---
uv run python eval_ablation_intent.py $RUN --out-dir $RUN/chapter6_4way --max-pairs 32

# --- P1: parity fix (5.3) first, then the full method sweep ----------------
uv run python 04_train_scoring_enhanced_dqn.py $RUN
uv run python 05_evaluate_methods.py $RUN
uv run python eval_multi_reward_comparison.py $RUN
uv run python 06_generate_figures.py $RUN

# --- P0/P1: new code ------------------------------------------------------
# 6.1 zero-shot interpolation, 5.4 held-out pairs, 5.5 seeds + 6.4 significance
```

**Out of scope for this repo:** the Chapter 1–3 `\gap{}`s (TOC structure, "trust" terminology, missing
BRITE/ECMP/SCION background) are writing tasks with no experiment behind them.
