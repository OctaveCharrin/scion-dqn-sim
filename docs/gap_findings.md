# Chapter 4 gap-closing results — findings for the thesis

**Produced:** 2026-07-25, in `/home/octav/fork-scion-dqn-sim`.
**Work order:** `docs/gap_experiments.md`.
**Audience:** the agent writing `/home/octav/thesis-report`. Nothing in that directory was modified.

All new artifacts live under `evaluation/run_20260722_180329/gap/` (plus `…/seeds/`). Figures are
copied to `docs/gap_results/figures/` with stable names. **The published artifact directory
`run_20260722_180329/chapter6_20260722_200655/` and all four original `.pth` checkpoints were verified
byte-identical (md5) after every run** — nothing the current draft cites has moved.

---

## 0. Read this first: harness validation

The ablation harness was rewritten to evaluate six methods instead of three (details in §7). Before any
new number is trusted, the rewritten harness was checked against the published one on the three methods
they share. It reproduces **every published value exactly**:

| | Throughput | Low-Latency | Low-Loss | Balanced | Adaptivity |
|---|---|---|---|---|---|
| Flat DQN (published / new) | 0.3330 / 0.3330 | 0.7009 / 0.7009 | 0.8678 / 0.8678 | 0.5614 / 0.5614 | 0.0107 / 0.0107 |
| Value-Concat (published / new) | 0.9788 / 0.9788 | 0.7082 / 0.7082 | 0.9373 / 0.9373 | 0.8942 / 0.8942 | 0.0502 / 0.0502 |
| Conditional-FiLM (published / new) | 0.9802 / 0.9802 | 0.7316 / 0.7316 | 0.9388 / 0.9388 | 0.8939 / 0.8939 | 0.1284 / 0.1284 |

A separate 256-case equivalence check confirms the new fast path and the original
`apply_action` path give bit-identical rewards (max abs difference 0.0).

**So: every existing number in Chapter 4 stands. What follows adds rows, controls and bounds around them.**

---

## 1. Results that contradict or narrow the current draft

These are the four findings the thesis author most needs to act on.

### 1.1 The Two-Stream-Concat control ties or beats FiLM everywhere — "modulation is required" is not supportable

> **Resolved 2026-07-26.** The thesis dropped FiLM entirely and now ships Two-Stream-Concat. See
> `gap_narrative.md` §3.4 for the decision, the re-anchoring work, and the claim-by-claim diff.

`\gap{}` at `sections/4_single_path.tex:552`.
**Artifact:** `evaluation/run_20260722_180329/gap/ablation_main/{ablation_reward_matrix.csv, table_6_1.tex}`

Full six-method table, 10 752 held-out decision contexts, single seed (identical contexts to the published run):

| Method | Throughput | Low-Latency | Low-Loss | Balanced | Adaptivity | Entropy (bits) |
|---|---|---|---|---|---|---|
| Flat DQN | 0.3330 | 0.7009 | 0.8678 | 0.5614 | 0.0107 | 0.0287 |
| Scoring DQN (unconditioned) | 0.9787 | 0.6792 | 0.9371 | 0.8978 | 0.0032 | 0.0085 |
| Value-Concat | 0.9788 | 0.7082 | 0.9373 | 0.8942 | 0.0502 | 0.1298 |
| **Two-Stream-Concat** | **0.9860** | 0.7312 | **0.9398** | **0.8974** | **0.1491** | **0.3745** |
| Conditional-FiLM | 0.9802 | **0.7316** | 0.9388 | 0.8939 | 0.1284 | 0.3139 |
| *Oracle (upper bound)* | *0.9861* | *0.7330* | *0.9409* | *0.8979* | *0.1609* | *0.4192* |

Two-Stream-Concat **wins on Throughput, Low-Loss and Balanced**, and loses to FiLM on Low-Latency by
**0.0004** — the intent the chapter's whole argument turns on. Its adaptivity (0.1491) is
**16 % above FiLM's** (0.1284). *(§2.11 repeats this over five training seeds and finds the two
statistically indistinguishable — so the single-seed ordering above should not be relied on in either
direction; what survives is that they are tied.)* Paired Wilcoxon over the shared contexts
(`gap/stats/significance.json`) makes all of these statistically significant given n = 10 752, which is
precisely why effect size matters more than the p-value here:

| Intent | Two-Stream − FiLM | p |
|---|---|---|
| Throughput | **+0.0057** | 9.5e−15 |
| Low-Loss | **+0.0009** | 1.6e−16 |
| Balanced | **+0.0035** | 1.4e−16 |
| Low-Latency | **−0.0005** | 3.0e−03 |

**Parameter counts (measured from the checkpoints, for Table 4.1's `Params` column):**

| Method | Q-network parameters |
|---|---|
| Flat DQN | 85 343 (85.3 k) |
| Scoring DQN (uncond.) | 36 226 (36.2 k) |
| Value-Concat | 36 866 (36.9 k) |
| Two-Stream-Concat | 37 506 (37.5 k) |
| Conditional-FiLM | 38 800 (38.8 k) |

The thesis's "1 934 additional parameters" for FiLM over Value-Concat is confirmed exactly. But note
what this does to the parsimony argument: **FiLM is 1 294 parameters *larger* than Two-Stream-Concat**,
which reaches the advantage stream simply by feeding it the full 12-D global vector. So FiLM is the more
expensive model *and* it does not outperform the cheaper control.

**Recommended framing.** The defensible claim is *not* that modulation is required, and it cannot rest
on parsimony either. It is:
> Conditioning that enters only path-independent terms provably cannot re-rank (Prop. 4.1), and the
> measurement confirms it: Value-Concat's adaptivity is 0.050 against 0.13–0.15 for both variants whose
> conditioning reaches the advantage stream. What matters is *that* the intent reaches the per-path
> comparison, not *how*: FiLM modulation and plain two-stream concatenation are practically tied on
> reward, and concatenation is slightly ahead on adaptivity (0.149 vs 0.128) at 1 294 fewer parameters.

The 2.5× adaptivity claim should be restated as *FiLM vs the value-stream variant*, with Two-Stream
reported as the honest control. If the thesis wants to keep FiLM as the shipped mechanism, the argument
available on this evidence is that its conditioning surface is *explicit and separable* (a generator
that produces a per-feature affine map, independent of the number of paths) rather than that it is
smaller or better — and that should be said as a design preference, not as an empirical result.

### 1.2 The ~44 % goodput gap belongs to per-path scoring, not to conditioning

`\gap{}` at `sections/4_single_path.tex:329`.
**Artifact:** `gap/ablation_main/ablation_reward_matrix.csv`, checkpoint `dqn_scoring_enhanced_model.pth` (newly trained).

The architectural control the chapter lacked now exists. The **unconditioned** enhanced path-scoring
DQN — which never sees an intent vector at all — achieves:

* goodput **8 210 Mbps**, against the conditional agents' 8 148–8 241 Mbps and the flat DQN's 5 623 Mbps;
* reward **0.9787 / 0.9371 / 0.8978** on Throughput / Low-Loss / Balanced — statistically
  indistinguishable from, and on **Balanced actually better than**, FiLM (0.8978 vs 0.8939, p = 4.8e−32).

Its chosen latency is flat at 17.02–17.06 ms across all four intents and its adaptivity is 0.0032, i.e.
it is exactly the intent-blind agent it should be.

**What conditioning actually buys** is therefore one thing, cleanly isolated: on **Low-Latency** — the
only evaluated intent that must steer away from the high-bandwidth default — FiLM scores 0.7316 against
the unconditioned scorer's 0.6792, **+0.0524** (p ≈ 0), cutting chosen latency from 17.02 ms to 12.84 ms
at a cost of ~900 Mbps.

**Recommended framing.** Split the ladder's claim in two: *per-path scoring* buys the 44 % goodput jump
over the flat DQN; *conditioning* buys per-intent re-ranking, worth ~0.05 reward on the one intent that
conflicts with the bandwidth-maximizing default. Both are real; only the second is about conditioning.

### 1.3 An oracle now exists, and FiLM is genuinely near-optimal — "near-oracle" can be restored

`\gap{}` at `sections/4_single_path.tex:398`.
**Artifact:** `gap/ablation_main/ablation_reward_matrix.csv`, column `optimality_gap`.

The oracle is the per-context `argmax` of the *actual reward under the intent being scored*, charged one
full probe so it is comparable to FiLM, and normalized identically (per-hour, per-pair best candidate).
It is implemented inside `run_ablation`, so it shares the exact contexts with every agent.

Optimality gap (oracle reward − method reward):

| Method | Throughput | Low-Latency | Low-Loss | Balanced |
|---|---|---|---|---|
| Two-Stream-Concat | **0.0002** | 0.0018 | 0.0012 | 0.0005 |
| Conditional-FiLM | 0.0059 | **0.0014** | 0.0021 | 0.0040 |
| Value-Concat | 0.0073 | 0.0248 | 0.0036 | 0.0037 |
| Scoring DQN (uncond.) | 0.0074 | 0.0538 | 0.0038 | 0.0001 |
| Flat DQN | 0.6532 | 0.0322 | 0.0731 | 0.3365 |

Relative to the oracle's own reward, FiLM's shortfall is **0.19 % (Low-Latency), 0.22 % (Low-Loss),
0.45 % (Balanced), 0.60 % (Throughput)**. The word "near-oracle" is now supported by an actual oracle
and can be reinstated — stated as *"within 0.6 % of a per-context reward oracle on every evaluated
intent"* — and the oracle should be added as the bottom row of Table 4.1 so the table reads as a
normalized optimality gap.

Note the oracle's own adaptivity is **0.1609** — so 0.16, not 1.0, is the ceiling for the Adaptivity
metric in this environment. FiLM at 0.1284 attains **80 %** of the achievable re-ranking, and
Two-Stream 93 %. That reframes the adaptivity numbers, which otherwise look small on a [0,1] scale.

### 1.4 The `w_probe` exclusion was correct, and can now be justified with a measured effect size

`\gap{}` at `sections/4_single_path.tex:464`. **The work order predicted the opposite result; the data
does not support it.**
**Artifact:** `gap/ablation_all6/ablation_reward_matrix.csv` (all six trained profiles as intents).

The work order argued the exclusion of `probe_minimal` / `probe_averse` was wrong because probe cost is
charged per hop, giving a probe-frugal intent a real incentive to prefer short paths. That incentive
exists but is negligible. Mean chosen hop count:

| Method | probe_minimal (w_probe=0.001) | probe_averse (w_probe=0.35) | Δ | for scale: Throughput → Low-Latency |
|---|---|---|---|---|
| **Oracle** | 1.899 | 1.881 | **−0.018** | 1.930 → 1.597 (**−0.333**) |
| Conditional-FiLM | 1.807 | 1.747 | −0.060 | 1.843 → 1.588 (−0.255) |
| Two-Stream-Concat | 1.851 | 1.856 | +0.005 | 1.938 → 1.572 (−0.366) |

Even the **optimal** policy shifts hop count by only 0.018 hops (≈1 %) across a 350× change in
`w_probe`, versus 0.333 hops between the two genuinely conflicting intents. `w_probe` therefore has
roughly **5 % of the leverage** of a real intent dimension.

**Independent corroboration** from a second harness and a different profile set
(`multi_reward_comparison.json`, the six legacy `REWARD_PROFILES`): between `low_probe_cost`
(w_probe = 0.005) and `high_probe_cost` (w_probe = 0.25) — a 50× change — the conditional agent's mean
probe cost per selection moves from **136.10 ms to 134.95 ms**, i.e. **1.15 ms, or 0.8 %**, and its
chosen latency from 16.76 to 16.04 ms. The unconditioned scoring agent is bit-identical across the two
(136.69 ms), as it must be. Same conclusion, reached independently.

**Recommended framing.** Keep the exclusion, and replace the incorrect justification
("path-independent") with the correct, measured one: *w_probe is charged per hop and so is not strictly
path-independent, but its leverage over the choice is ~5 % of that of the throughput/latency axis — the
per-context reward oracle itself moves the chosen hop count by only 0.018 hops across the full w_probe
range, so these two profiles cannot discriminate between paths and are excluded from the conditioning
studies.* This also answers the "they consume a third of the training schedule" concern: they do, and
they are near-duplicates of Balanced by construction.

---

## 2. Gaps closed with supporting (non-contradicting) results

### 2.1 Zero-shot generalization across intents — the claim holds

`\gap{}` at `sections/4_single_path.tex:653` (missing §4.5.4), `:274`, `sections/1_introduction.tex:61`,
`sections/6_synthesis.tex:134`. **This was the chapter's most conspicuous hole.**
**Artifacts:** `gap/zeroshot/{intent_interpolation.csv, intent_interpolation_summary.json}`,
`gap/zeroshot_heldout/…`, figure `docs/gap_results/figures/fig_intent_interpolation.png`.

New script `evaluation/eval_intent_interpolation.py` sweeps
`w(t) = (1−t)·bandwidth_max + t·delay_averse` over `t ∈ {−0.2, −0.1, 0.0, 0.1, …, 1.0, 1.1, 1.2}`.
**Only `t = 0` and `t = 1` are weight vectors any agent has ever seen**; all 13 other points are unseen,
and the two outside [0,1] are extrapolations beyond both trained profiles. Same 32 pairs × 336 held-out
hours = 10 752 contexts per point.

| Method | chosen latency at t=0 | at t=1 | monotone? | Spearman ρ(latency, t) | largest single step |
|---|---|---|---|---|---|
| Conditional-FiLM | 17.31 ms | 12.84 ms | **yes** | **−1.000** | 14.8 % of span |
| Two-Stream-Concat | 18.40 ms | 12.73 ms | **yes** | **−1.000** | 24.7 % of span |
| Value-Concat | 17.07 ms | 15.11 ms | **yes** | **−1.000** | 17.1 % of span |

The curve is strictly monotone with perfect rank correlation, and no single step carries more than a
quarter of the total change — the signature of a learned mapping, not six memorized points (a step
function would put ~100 % of the span in one step). The endpoints reproduce the trained-profile
behaviour exactly (17.31 / 12.84 ms match the ablation's Throughput / Low-Latency rows).

Note Value-Concat interpolates smoothly too, but over a **2.0 ms** range against FiLM's **4.5 ms** and
Two-Stream's **5.7 ms** — consistent with Prop. 4.1: it can only move behaviour through indirect
training effects, so it traverses less of the behaviour space.

**The same sweep on the genuinely held-out pairs of §2.3** (`gap/zeroshot_heldout/`) — 32 pairs across
32 source ASes the agents never trained on — reproduces the result:

| Method | latency at t=0 | at t=1 | monotone? | ρ | largest step |
|---|---|---|---|---|---|
| Conditional-FiLM | 24.06 ms | 19.47 ms | yes | −1.0000 | 17.1 % |
| Two-Stream-Concat | 24.76 ms | 19.16 ms | yes | −1.0000 | 22.2 % |
| Value-Concat | 23.55 ms | 21.07 ms | yes | −1.0000 | 21.5 % |

This is **unseen intents on unseen pairs simultaneously**, and it is the strongest single piece of
generalization evidence produced by this work. It supports both the *Zero-Shot Generalization Across
Objectives* and (partially) the *Structural & Topological Generalization* properties of
`sec:problem:properties`.

**Recommended framing.** The word "zero-shot" can be restored to the abstract and
`sec:intro:contributions`, supported by this experiment, and §4.5.4 can be written. State the scope
honestly: interpolation *and* modest extrapolation along the segment between two trained profiles, on
one topology.

### 2.2 Variable path sets and order invariance — Contribution 2 is now measured

`\gap{}` at `sections/4_single_path.tex:672` (missing §4.5.5), `sections/6_synthesis.tex:116`.
**Artifacts:** `gap/pathcount/{pathcount_scaling.csv, order_invariance.json}`,
figure `docs/gap_results/figures/fig_pathcount_scaling.png`.

Binning by natural `N` is uninformative (7 988 of 8 010 pairs have exactly 30 paths), so
`evaluation/eval_pathcount_scaling.py` restricts the candidate set at decision time to
`N ∈ {2, 4, 8, 16, 30}`, identical subsets for every method, and reports regret against the best path
in that restricted set.

Mean regret against the best path in the restricted set (10 752 contexts, Balanced intent):

| Method | N=2 | N=4 | N=8 | N=16 | N=30 |
|---|---|---|---|---|---|
| Flat DQN | 0.2387 | 0.4162 | 0.4983 | 0.4816 | 0.3365 |
| Scoring DQN (uncond.) | 0.0055 | 0.0108 | 0.0110 | 0.0044 | **0.0001** |
| Value-Concat | 0.0045 | 0.0093 | 0.0102 | 0.0056 | 0.0037 |
| Two-Stream-Concat | 0.0074 | 0.0094 | 0.0056 | 0.0034 | 0.0005 |
| Conditional-FiLM | 0.0067 | 0.0109 | 0.0132 | 0.0081 | 0.0040 |

**One model, no retraining, no padding, across a 15× range in `N`: every scoring agent holds regret
between 0.0001 and 0.013 — two orders of magnitude below the flat DQN's 0.24–0.50**, which must pad to
a fixed 30-way action head and gets *worse* as `N` grows into the middle of its range.

Report one caveat honestly: the scoring agents are slightly *better* at `N = 30` than at small `N`
(0.0001–0.004 vs 0.005–0.013). They were trained exclusively on 30-candidate sets, and the per-path
`bw_ratio` feature is normalized by the maximum *within the presented set*, so a small subsample shifts
the observation distribution off-distribution. The effect is real but bounded at ~0.01 reward — the
architecture handles unseen `N` gracefully rather than perfectly.

**Order invariance is exact.** Re-scoring each of the 10 752 contexts under 3 random permutations of the
path ordering (**32 256 trials per agent**), all four scoring/conditional agents chose the *same
underlying path* in **100.0000 %** of trials. The flat DQN is excluded by construction — its action is
an index into a padded vector, so it is not permutation-equivariant, which is exactly the architectural
point. A unit test now encodes the property
(`tests/test_chapter6_eval.py::test_scoring_nets_are_permutation_equivariant`).

An independent cross-check via a different route (`gap/conditional_sensitivity.json`,
`analyze_conditional_sensitivity.py` over 32 pairs × 24 hours) gives a cross-profile action-diversity
rate of **0.177** over six profiles, and shows the conditional agent agreeing with the unconditioned
scoring agent on **91.5 %** of decisions — consistent with §1.2's finding that conditioning changes
roughly one decision in twelve. *(Note that script uses the six legacy `REWARD_PROFILES`, not the
distinctive set the agents trained on, so it is also a mild off-distribution probe.)*

### 2.3 Genuinely held-out pairs — and a correction to the premise

`\gap{}` at `sections/4_single_path.tex:448`, `sections/3_problem.tex:26`, `sections/6_synthesis.tex:116`.
**Artifacts:** `run_20260722_180329/heldout_pairs.json`, `gap/ablation_heldout/`, `gap/zeroshot_heldout/`.

**Correction to the work order and the thesis.** Both assume training covers `pair_pool[:64]` via
`DQN_TRAIN_PAIR_CAP`. It does not. `DQN_TRAIN_PAIR_CAP` only sizes the *episode budget*
(`src/rl/path_selection_train.py:52-73`); each episode samples a pair uniformly from the **whole
8 010-pair pool** (`path_selection_train.py:570`, `:191`). Replaying `random.Random(123)` for the
episode counts actually trained shows the agents ever visited exactly **644 distinct pairs**, and only
**2 of the 32 evaluation pairs** are among them.

That makes an exactly-disjoint held-out set constructible rather than approximate.
`evaluation/build_heldout_pairs.py` replays the RNG, subtracts, and samples **32 never-trained pairs
spanning 32 distinct source ASes** (the current 32 all share source AS 0) from the 7 366 unseen pairs.

| Method | Throughput | Low-Latency | Low-Loss | Balanced | Adaptivity |
|---|---|---|---|---|---|
| Flat DQN | **−0.0848** | 0.5167 | 0.6521 | 0.2461 | 0.0078 |
| Scoring DQN (uncond.) | 0.9780 | 0.5571 | 0.8869 | 0.8570 | 0.0104 |
| Value-Concat | 0.9797 | 0.5923 | 0.8878 | 0.8539 | 0.0704 |
| Two-Stream-Concat | 0.9842 | 0.6065 | 0.8887 | 0.8572 | **0.1903** |
| Conditional-FiLM | 0.9822 | 0.6062 | 0.8890 | 0.8559 | 0.1514 |
| *Oracle* | *0.9845* | *0.6099* | *0.8919* | *0.8582* | *0.1806* |

Three things to report:

1. **The scoring/conditional agents generalize structurally.** Absolute rewards drop on Low-Latency
   (0.732 → 0.606) — **but so does the oracle** (0.733 → 0.610). FiLM's optimality gap is 0.0014
   in-pool and 0.0037 held-out: the drop is a property of *these pairs' available headroom*, not a
   generalization failure. Adaptivity actually **rises** (0.128 → 0.151 for FiLM, 0.149 → 0.190 for
   Two-Stream).
2. **The flat DQN does not generalize at all** — Throughput reward goes *negative* (0.333 → −0.0848).
   Its fixed, index-addressed action space is tied to the pairs it trained on. This is direct evidence
   for the *Structural & Topological Generalization* property and for abandoning the flat architecture.
3. The wording "held-out pairs" can now be used accurately — for these tables. The main tables remain
   temporal-only and should still say so.

### 2.4 Inference latency — the deployability claim is supported

`\gap{}` at `sections/6_synthesis.tex:103`.
**Artifact:** `run_20260722_180329/evaluation_results.json` (newly produced; step 05 had never been run
on this run directory).

`avg_selection_time_ms` as previously recorded spans `reset → step` and includes the simulator's whole
per-path probing loop, so it could not answer this gap. A separate `avg_inference_time_ms` now times
**only** the `agent.act(...)` / `selector.select_path(...)` call:

| Method | mean inference | p95 | p99 |
|---|---|---|---|
| Conditional-FiLM (`conditional_dqn`) | **0.482 ms** | 1.027 ms | 2.203 ms |
| Scoring DQN (enhanced) | 0.397 ms | 0.850 ms | 1.925 ms |
| Scoring DQN (simple) | 0.310 ms | 0.658 ms | 2.104 ms |
| Flat DQN | 0.257 ms | 0.500 ms | 1.054 ms |
| Heuristics (all) | 0.008–0.023 ms | ≤0.046 ms | ≤0.091 ms |

Measured on CPU only (22-core WSL2 host, no GPU), scoring 30 candidate paths. The conditioned selector
decides in **under half a millisecond**, p99 2.2 ms — three orders of magnitude inside a
path-selection budget of seconds-to-minutes, and comfortably inside a 33 ms video frame. The
`sec:analysis:deployability` claim is supported for the single-path agent. *(The multipath transport
agent's per-frame budget is a Chapter 5 question and is not covered here.)*

### 2.5 Training curves and convergence

`\gap{}` at `sections/4_single_path.tex:349`, `sections/appendix.tex:86`.
**Artifacts:** `gap/training/{training_summary.json, figures/fig_training_curves.png,
figures/fig_training_per_intent_*.png}`.

| Agent | Episodes | First-50 mean | Last-50 mean | Gain | Loss trend | Final ε |
|---|---|---|---|---|---|---|
| Flat DQN | 533 | −0.216 | **+0.049** | +0.265 | **+0.046 (rising)** | 0.0691 |
| Scoring DQN (uncond.) | 533 | −0.160 | +0.809 | +0.969 | +0.015 | 0.0691 |
| Scoring DQN (simple) | 533 | −0.153 | +0.815 | +0.968 | +2.990 | 0.0691 |
| Value-Concat | 666 | −0.051 | +0.816 | +0.867 | +0.022 | 0.0355 |
| Two-Stream-Concat | 666 | −0.035 | +0.814 | +0.849 | +0.003 | 0.0355 |
| Conditional-FiLM | 666 | −0.057 | **+0.822** | +0.878 | **−0.018 (falling)** | 0.0355 |

These match the values quoted in the work order exactly, which validates the script.

The `\gap{}` at `:349` asks whether the flat DQN is undertrained rather than architecturally limited.
The answer is now sharper than "maybe": the flat DQN plateaus at **+0.049** with a *rising* loss, while
the **unconditioned scoring agent trained for the identical 533 episodes on the identical data reaches
+0.809**. Same budget, same data, same optimizer — 16× the final reward. The flat agent's deficit is
therefore attributable to its architecture (a fixed 30-way action head over a 5-D global state with no
per-path features), not to the training budget. That is a stronger statement than the chapter currently
makes and it removes the caveat the `\gap{}` was worried about.

Per-intent curves (`fig_training_per_intent_film.png` and the two concat variants) split the 666
episodes by the stratified schedule's `episode_weight_profiles`, giving ~111 episodes per intent —
the only place `probe_minimal` and `probe_averse` appear at all.

ε reaches only 0.0355 (conditional) / 0.0691 (flat), never the stated 0.01 floor — as the appendix
`\gap{}` at `:68` notes.

### 2.6 Reward-vs-congestion: the two-panel ceiling figure

`\gap{}` at `sections/4_single_path.tex:778`.
**Artifact:** `gap/ceiling_two_panel/figures/fig_6_3_ceiling.png` →
`docs/gap_results/figures/fig_ceiling_two_panel.png`.

`plot_ceiling_vs_congestion` now accepts `metric="both"` and renders goodput and reward side by side
with the per-bin decision count annotated (1 792 decisions per bin per method). The figure is generated
from **the exact CSV the current draft cites** (copied, not regenerated), so the numbers in the prose
at `sec:p1eval:ceiling` are unchanged: FiLM's reward is flat at 0.896 → 0.886 while its goodput falls
9 067 → 6 844 Mbps (−24.5 %), and shortest-path collapses 0.72 → 0.31 with random going 0.26 → −0.32.

Regenerate with:
`uv run python 06b_generate_chapter6_figures.py <run> --artifact-dir <dir> --metric both`

### 2.7 Environment realism — the statistic that settles §4.1.1

`\gap{}` at `sections/4_single_path.tex:49`, `:68`, `:103`.
**Artifact:** `gap/realism/pair_selection_spread.json`, `evaluation/analyze_pair_selection_spread.py`.

The chapter asserts on both sides — that prior work's environment lacked the fidelity to make path
selection matter, and that this one has it — without a number. The quantity that settles it is the
spread of achievable goodput across a pair's candidate paths *within one decision context*.

Over all 10 752 held-out decision contexts (32 pairs × 336 hours, every pair having exactly 30
candidates):

| Statistic | Value |
|---|---|
| Best candidate's goodput ÷ median candidate's | **3.04×** (p10 1.57, p90 **18.26**) |
| Best − worst candidate goodput | **7 983 Mbps** mean (best itself averages 8 250 Mbps) |
| Coefficient of variation of goodput across candidates | **1.43** median |
| Worst − best candidate latency | **29.3 ms** median (ratio **3.5×**) |
| Identity of the best-goodput path changes hour-to-hour | **8.6 %** of consecutive hours |
| Contexts with no usable bandwidth on any path | 0.09 % |

**Path choice is a first-order effect in this environment, and a time-varying one.** The best path
carries three times the median path's goodput and effectively all of it relative to the worst; and the
right answer changes roughly every twelve hours. That is the number §4.1.1 needs, and it makes the
argument the chapter wants to make.

The same script bounds how far the intents *can* diverge, by asking how often the per-metric optima are
the same path:

| Coincidence | Fraction of contexts |
|---|---|
| argmax-bandwidth = argmin-latency | **40.9 %** |
| argmax-bandwidth = argmin-loss | **46.8 %** |
| argmin-latency = argmin-loss | **82.8 %** |
| all three the same path | 40.8 % |

So a genuine bandwidth/latency conflict exists in **59 %** of decisions — conditioning has real room to
work on that axis, which is exactly where it pays off (§1.2). But **latency-optimal and loss-optimal are
the same path 83 % of the time**, which is the structural reason the Low-Loss intent has nothing of its
own to do. See §3.7.

### 2.8 Per-intent probing — a strictly stronger claim than the chapter currently makes

`\gap{}` at `sections/4_single_path.tex:724`.
**Artifacts:** `gap/probing/{probing_quality_by_intent.csv, probing_quality.csv, ceiling_by_congestion.csv}`,
figure `docs/gap_results/figures/fig_probing_by_intent.png`.

`run_probing_ceiling` is now parameterized by intent (it previously hard-coded `balanced_extreme`, the
one intent where a single heuristic is close to the right rule). Reward per (method, intent), all on the
same 10 752 contexts:

| Method | probes/sel | probe ms/sel | Throughput | Low-Latency | Low-Loss | Balanced |
|---|---|---|---|---|---|---|
| **Conditional-FiLM** | **1** | **132–137** | 0.980 | 0.732 | **0.939** | **0.894** |
| Widest-Path | 30 | 5 388 | **0.982** | 0.652 | 0.930 | 0.891 |
| Lowest-Latency | 30 | 360 | 0.449 | **0.733** | 0.899 | 0.629 |
| ECMP | 30 | 360 | 0.577 | 0.711 | 0.912 | 0.693 |
| SCION-Default | 30 | 360 | 0.417 | 0.726 | 0.893 | 0.610 |
| Shortest-Path | 30 | 360 | 0.383 | 0.721 | 0.888 | 0.590 |
| Random | 30 | 360 | −0.569 | 0.247 | 0.558 | −0.088 |

This is the claim `sec:disc:interpretation` already asserts, now measured:

> **One conditioned policy tracks the per-intent strongest heuristic — and the strongest heuristic
> changes with the intent.** Widest-path wins under Throughput (0.982 vs FiLM's 0.980, a 0.002 gap) and
> lowest-latency wins under Low-Latency (0.733 vs 0.732, a 0.001 gap); FiLM *beats every heuristic* under
> Low-Loss and Balanced. Meanwhile each heuristic collapses under the other's objective — widest-path
> falls to 0.652 under Low-Latency, lowest-latency to 0.449 under Throughput. FiLM does this with
> **1 probe instead of 30**, at 132–137 ms against 360 ms for the latency-probing heuristics and
> 5 388 ms for widest-path — i.e. **39× cheaper in probe time than the only method that beats it on any
> intent, and 2.7× cheaper than the cheapest heuristic.**

Report probes per selection alongside milliseconds, as the `\gap{}` asks: "1 versus 30" is the legible
quantity, and note FiLM is the *cheapest* method overall in both.

### 2.9 Why two of the four intents are indistinguishable — a measured answer

`\gap{}` at `sections/4_single_path.tex:609` and `:643` (the boxplot that contradicts its own caption).
**Artifacts:** `gap/intent/intent_selection_metrics.csv` (now carries `loss_rate`, `hop_count`,
`trust_own_w`), figure `docs/gap_results/figures/fig_intent_boxplots_corrected.png`.

The chapter explains the Low-Loss/Balanced degeneracy by saying low-loss paths "largely coincide with"
high-bandwidth paths. The measurement gives a sharper reason: **there is almost no loss in this
environment to avoid.** Across all 43 008 FiLM selections, **98.85 % of chosen paths have exactly zero
loss** (mean 0.00072, p90 = 0). Per intent:

| Intent | mean chosen latency | mean chosen goodput | selections with loss > 0 | canonical trust |
|---|---|---|---|---|
| Throughput | 17.31 ms | 8 215 Mbps | 0.48 % | 0.9134 |
| Low-Latency | **12.84 ms** | 7 307 Mbps | **3.15 %** | **0.9346** |
| Low-Loss | 16.25 ms | 8 155 Mbps | 0.48 % | 0.9187 |
| Balanced | 16.13 ms | 8 148 Mbps | 0.48 % | 0.9193 |

The Low-Loss intent achieves **exactly the same loss exposure as Throughput and Balanced** (0.48 %),
because that is already the floor. The *Low-Latency* intent is the only one that moves loss — and it
moves it **up**, to 3.15 %, buying 4.5 ms of latency at the cost of more lossy short paths. Combined
with §2.7's finding that latency-optimal and loss-optimal coincide 82.8 % of the time, the conclusion is
structural rather than a failure of the mechanism.

**Figure fix.** The boxplot's third panel was titled "Low-Loss intent → higher" while its data show
Low-Latency winning on canonical trust. The figure is regenerated with (a) the trust panel honestly
annotated "Low-Latency intent → higher", which is what the canonical `w3 = w4 = 0.5` definition implies,
and (b) a fourth panel showing the **percentage of selections incurring any loss**, since a loss boxplot
is a flat line at zero and hides the finding. Every annotation now matches its data.

*(A `trust_own_w` column — trust scored under each intent's own `w3, w4` — is also emitted, but it is
**not** a usable cross-intent comparison: Throughput's `w3 = w4 = 0.05` makes its trust trivially ≈0.99.
Recommend reporting canonical trust plus the loss-exposure bar, not `trust_own_w`.)*

**Recommended framing.** State that Intent Alignment is demonstrated on the **goodput↔latency axis**,
which is the only axis this environment offers real conflict on (59 % of contexts), and say why the
loss axis is degenerate here (98.9 % zero-loss selections; 7.8 % of links oversubscribed is not enough
to make loss a discriminating objective). That is a defensible, measured scoping of the claim.

### 2.10 Can a better choice of intents fix it? No — and the oracle proves it without a retrain

The `\gap{}` at `:609` asks for one of the degenerate intents to be *replaced* by "a profile whose
optimum genuinely conflicts with high bandwidth — a hop-count-averse or trust-dominant intent". We
tested that proposal directly.

The reward (`evaluation_env.py:323-330`) exposes only three levers: goodput, `trust(loss, delay)`, and a
probe penalty. Hop count enters **only** through the probe charge (`100 + 20·hops`). So two candidate
replacements were defined (`DQN_CONDITIONAL_PROFILES=conflicting` in `src/rl/reward_profiles.py`):

* `probe_extreme` — `w_probe = 0.8`, i.e. **16× the `probe_averse` value**, the strongest hop-aversion
  the reward can express;
* `loss_only` — `w1=0.05, w2=0.95, w3=1.0, w4=0.0`: loss avoidance with the latency term removed
  entirely, so it cannot ride on the latency axis the way `loss_averse` does.

Rather than retrain first, we evaluated the **oracle** under them (`gap/ablation_conflicting/`,
2 688 contexts). The oracle is the per-context reward `argmax`, so it bounds what *any* policy could
achieve — trained, untrained or perfect:

| Intent (oracle row) | mean hops | mean latency | mean goodput | reward |
|---|---|---|---|---|
| bandwidth_max | 1.932 | 18.33 ms | 8 260 Mbps | 0.986 |
| delay_averse | **1.592** | **12.48 ms** | 7 140 Mbps | 0.734 |
| balanced_extreme | 1.849 | 17.21 ms | 8 236 Mbps | 0.898 |
| **probe_extreme** (w_probe = 0.8) | **1.858** | 17.39 ms | 8 241 Mbps | 0.725 |
| **loss_only** | **1.911** | 18.10 ms | 8 258 Mbps | 0.986 |
| probe_minimal | 1.902 | 17.91 ms | 8 255 Mbps | 0.945 |

Both proposals fail, for different reasons:

* **`probe_extreme` moves the optimal hop count by 0.009 hops** relative to `balanced_extreme`
  (1.858 vs 1.849). At sixteen times the probe weight, the hop axis is still dead. The per-hop probe
  charge is simply too small next to the goodput term for any weighting to matter.
* **`loss_only` collapses onto `bandwidth_max`** — identical reward (0.986), goodput within 2 Mbps,
  hops within 0.02. With loss ≈ 0 everywhere, deleting the latency term leaves nothing but goodput, so
  a "pure loss" intent *is* a throughput intent in this environment.

**Conclusion, and why no retrain was run.** No reweighting of this reward function can produce a fourth
discriminating intent, because the constraint is the *environment*, not the profile set — and the oracle
bound applies to every policy, so training an agent on these profiles could not have changed the answer.
This is a stronger and cheaper result than a retrain would have given.

**Recommended framing.** Do not present this as a bad choice of intents that better weights would fix.
The honest statement is that this environment supports exactly **one** behavioural trade-off axis
(goodput ↔ latency), that conditioning is demonstrated on it, and that making loss or hop count
discriminating requires changing the *reward or the environment* — e.g. adding an explicit hop-count
term, or raising offered load until loss is common (only 7.8 % of links are oversubscribed today; see
`simulation_metadata.json`). That is a concrete, actionable limitation to state in Future Work.

The `conflicting` profile set is committed and selectable, so the experiment is one env-var away if the
reward is ever changed:
`DQN_CONDITIONAL_PROFILES=conflicting uv run python 04_train_conditional_dqn.py $RUN`.

### 2.11 Seed variance — the central claim survives, the secondary one does not

`\gap{}` at `sections/4_single_path.tex:566` and `sections/appendix.tex:68`.
**Artifacts:** `gap/stats/seed_variance.json`, `run_20260722_180329/seeds/seed{1..5}/`,
`evaluation/run_seed_sweep.sh`.

Nothing in the repo previously seeded torch or numpy, so runs were not reproducible at all.
`src/rl/seeding.py::set_global_seeds` now fixes network init, prioritized-replay sampling and
ε-greedy exploration, exposed as `--seed` / `DQN_SEED`. The environment's pair, hour and
profile RNGs are deliberately **left fixed** (42/123/456/789), so every seed sees the identical stream
of training contexts and only the learning process varies — which is what "seed variance" should mean
here, and it keeps the never-trained held-out set of §2.3 valid.

All three conditional variants retrained under seeds 1–5 (15 trainings) and re-evaluated on the same
10 752 contexts. Mean ± 95 % CI over seeds:

| Method | Throughput | Low-Latency | Low-Loss | Balanced | Adaptivity |
|---|---|---|---|---|---|
| Value-Concat | 0.9760 [0.9722, 0.9798] | 0.7058 [0.6996, 0.7119] | 0.9382 [0.9374, 0.9390] | 0.8923 [0.8877, 0.8970] | 0.0425 [0.0309, 0.0541] |
| Two-Stream-Concat | 0.9839 [0.9815, 0.9864] | 0.7296 [0.7279, 0.7312] | 0.9393 [0.9368, 0.9418] | 0.8963 [0.8950, 0.8977] | 0.1398 [0.1293, 0.1503] |
| Conditional-FiLM | 0.9819 [0.9793, 0.9846] | 0.7308 [0.7302, 0.7313] | 0.9391 [0.9380, 0.9401] | 0.8958 [0.8951, 0.8965] | 0.1364 [0.1052, 0.1676] |
| *Oracle* | *0.9861* | *0.7330* | *0.9409* | *0.8979* | *0.1609* |

Three conclusions, and they are the point of the whole exercise:

1. **The chapter's load-bearing comparison is robust.** FiLM beats Value-Concat on Low-Latency
   0.7308 [0.7302, 0.7313] vs 0.7058 [0.6996, 0.7119] — **the confidence intervals are disjoint by a
   wide margin**. Adaptivity likewise: 0.1364 [0.1052, 0.1676] vs 0.0425 [0.0309, 0.0541], disjoint.
   Proposition 4.1's empirical signature reproduces across every seed. **This claim can be stated with
   confidence intervals and it survives.**
2. **The FiLM-vs-Two-Stream ordering does not reproduce.** On adaptivity, FiLM [0.1052, 0.1676] and
   Two-Stream [0.1293, 0.1503] **overlap heavily**; the single-seed 0.128-vs-0.149 result is inside
   seed noise. On reward the two overlap on Low-Latency, Low-Loss and Balanced, and Two-Stream is
   marginally ahead on Throughput. They are statistically indistinguishable. This confirms §1.1 from a
   second direction: the mechanism choice is not what matters, and the chapter should not claim it is.
3. **FiLM is the less stable of the two.** Its adaptivity standard deviation across seeds is **0.0251
   against Two-Stream's 0.0084** — a 3× wider spread, and by far the widest CI in the table. If the
   thesis argues for FiLM it should not argue for it on reliability.

The appendix's hyperparameter table can now fill its "number of seeds" column with **5**, and the
per-agent wall-clock is ~1–3 min per conditional agent (666 episodes) on a 22-core CPU host.

---

## 3. Corrections to statements in the current draft

1. **`sections/4_single_path.tex:448`** — "held-out pairs": the reason the split is weak is not that
   training used a 64-pair prefix. Training samples uniformly from all 8 010 pairs and visits 644 of
   them; the 32 evaluation pairs are weak because they all share **source AS 0**, not because they
   were trained on (only 2 of 32 were). See §2.3.
2. **`sections/4_single_path.tex:464`** — the draft's `\gap{}` claims the `w_probe` exclusion is
   "stated more strongly than the environment supports". Measurement says the exclusion is right; see
   §1.4.
3. **Proposition 4.1's premise is violated by the environment.** Per-path feature index 6 is `trust`,
   computed from `w3` and `w4` (`src/simulation/evaluation_env.py:397`). Intent therefore *does* enter
   the per-path features, which is why Value-Concat's adaptivity is 0.0502 rather than exactly 0. The
   proposition holds for the *architecture* — the unit test at `tests/test_chapter6_eval.py:194`
   verifies the value-only network's argmax is exactly intent-invariant — but not for the deployed
   system. The chapter should state this leak explicitly; it is the difference between "cannot
   re-rank" and "re-ranks only through a single feature it does not control".
4. **ECMP was not doing equal-cost spreading.** In `05_evaluate_methods.py` the flow stub carried
   neither `source_as`, `destination_as` nor `flow_id`, so `ECMPSelector` hashed `(None, None, None)`
   and returned one constant index for every pair and every hour. Fixed (the stub now varies
   `flow_id` by hour). In `chapter6_eval.py` the stub does carry the AS pair but pins `flow_id = 0`,
   so ECMP there spreads across pairs but never across flows — it remains, as the `\gap{}` at `:416`
   says, "hash-pinned shortest path" and should be labelled as such in Chapter 4.
5. **`avg_selection_time_ms` is not inference latency** — see §2.4.
6. **Method label change:** new CSVs label `conditional_concat` as **"Value-Concat"** (was
   "Conditional-Concat"), matching the thesis's own wording, and add "Two-Stream-Concat",
   "Scoring DQN (uncond.)" and "Oracle".

---

## 4. Still open after this work

* **`sections/6_synthesis.tex:116` and `:134` (larger topologies).** Out of scope by agreement. Only
  one topology exists; evaluating an existing checkpoint against a *new* run directory needs a
  cross-run loader that does not exist. `EVAL_BRITE_N_NODES` makes generating a second topology cheap,
  but note two traps: `beacon_pipeline.py` switches to 200-pair sampling above 200 nodes, and
  `DQN_TRAIN_PAIR_CAP = 64` means the training budget does not grow with topology size. §2.2 and §2.3
  give partial evidence (path-count scaling and cross-source-AS pairs).
* **`sections/5_multipath.tex` gaps and NS-3 parity** — Chapter 5 work, untouched.
* **`sections/1_introduction.tex:25`, `sections/2_background.tex:36`/`:46`, `sections/3_problem.tex:26`,
  `sections/0_acknowledgements.tex`** — writing tasks with no experiment behind them.
* **`sections/appendix.tex:80` (SAC hyperparameters)** — Chapter 5.
* **Per-agent wall-clock training time** (`appendix.tex:68`) — the training loop still does not record
  it. Observed externally: ~1–3 min per agent on a 22-core CPU host, no GPU.

---

## 5. Artifact index

| What | Path |
|---|---|
| Six-method ablation + oracle | `evaluation/run_20260722_180329/gap/ablation_main/` |
| Same, genuinely held-out pairs | `…/gap/ablation_heldout/` |
| All six trained intents (incl. probe profiles) | `…/gap/ablation_all6/` |
| Candidate replacement intents + oracle bound | `…/gap/ablation_conflicting/` |
| Corrected intent-alignment metrics | `…/gap/intent/` |
| Cross-profile sensitivity cross-check | `…/gap/conditional_sensitivity.json` |
| Zero-shot intent interpolation | `…/gap/zeroshot/`, `…/gap/zeroshot_heldout/` |
| Path-count scaling + order invariance | `…/gap/pathcount/` |
| Per-intent probing / ceiling | `…/gap/probing/` |
| Two-panel ceiling figure (published data) | `…/gap/ceiling_two_panel/figures/` |
| Training curves | `…/gap/training/` |
| Environment realism / candidate spread | `…/gap/realism/` |
| Seed variance + paired significance | `…/gap/stats/`, `…/seeds/seed{1..5}/` |
| Held-out pair definition | `…/heldout_pairs.json` |
| Inference latency, full method sweep | `…/evaluation_results.json` |
| Multi-profile method comparison | `…/multi_reward_comparison.json` |
| Figures for the thesis | `docs/gap_results/figures/` |

### Figures

All are 300 dpi PNG, LNCS styling, Okabe-Ito colourblind-safe palette with redundant marker encoding.

| File in `docs/gap_results/figures/` | Replaces / adds | Thesis `\gap{}` |
|---|---|---|
| `fig_ceiling_two_panel.png` | two-panel version of `p1eval_ceiling.png` (goodput + reward, bin counts) | `4_single_path.tex:778` |
| `fig_intent_interpolation.png` | **new** — zero-shot intent sweep | `:653`, `:274` |
| `fig_pathcount_scaling.png` | **new** — regret vs candidate-set size | `:672` |
| `fig_probing_by_intent.png` | **new** — per-intent quality vs probes/selection | `:724` |
| `fig_intent_boxplots_corrected.png` | replaces `p1eval_intent_boxplots.png` (annotations now match data; loss panel added) | `:643`, `:609` |
| `fig_intent_heatmap.png` | regenerated reward heatmap (unchanged data) | — |
| `fig_training_curves.png` | **new** — episode reward + loss, all six agents | `:349`, `appendix.tex:86` |
| `fig_training_per_intent_{film,concat2stream,valueconcat}.png` | **new** — per-intent learning curves | `appendix.tex:86` |
| `fig_topology_{brite,scion,peering}.png` | the three unused BRITE→SCION→peering pipeline figures | `:68` |

The topology figures are the ones the work order suggested copying into the thesis: they show that BRITE
is run at 50 nodes with all links at exactly 10 Gbps and that the 90 ASes / 528 links with heterogeneous
capacity come from structure the *converter* adds. They are copied here rather than into the thesis
tree; the thesis author can move them.

> **Note:** `.gitignore` excludes `*.png` repo-wide, so these files exist on disk but will not appear in
> `git status`. Copy them out, or regenerate with the commands in §6.

---

## 6. Reproducing

```bash
cd /home/octav/fork-scion-dqn-sim/evaluation
export RUN=run_20260722_180329          # always pass the run dir positionally first:
                                        # resolve_run_dir() reads sys.argv[1] directly and
                                        # will treat a bare flag as the run directory.

uv run python 04_train_scoring_enhanced_dqn.py $RUN          # the architectural control
uv run python build_heldout_pairs.py $RUN                    # never-trained pair set

uv run python eval_ablation_intent.py $RUN --out-dir $RUN/gap/ablation_main --max-pairs 32
uv run python eval_ablation_intent.py $RUN --out-dir $RUN/gap/ablation_heldout \
    --max-pairs 32 --pairs-json $RUN/heldout_pairs.json
uv run python eval_ablation_intent.py $RUN --out-dir $RUN/gap/ablation_all6 --max-pairs 32 \
    --profiles bandwidth_max delay_averse loss_averse balanced_extreme probe_minimal probe_averse

uv run python eval_intent_interpolation.py $RUN --out-dir $RUN/gap/zeroshot --steps 11
uv run python eval_pathcount_scaling.py    $RUN --out-dir $RUN/gap/pathcount --permutations 3
uv run python eval_probing_ceiling.py      $RUN --out-dir $RUN/gap/probing \
    --profiles balanced_extreme bandwidth_max delay_averse loss_averse
uv run python analyze_pair_selection_spread.py $RUN --out-dir $RUN/gap/realism
uv run python plot_training_curves.py      $RUN --out-dir $RUN/gap/training
uv run python 05_evaluate_methods.py       $RUN                      # inference latency
./run_seed_sweep.sh $RUN 1 2 3 4 5                                   # seed variance
uv run python analyze_seed_variance.py     $RUN --out-dir $RUN/gap/stats
```

---

## 7. Code changes

All additive; `uv run python -m pytest` is green (82 passed) and `flake8` reports exactly the same
247 pre-existing findings as before the work (no new lint).

**Modified**

| File | Change |
|---|---|
| `src/pipeline/chapter6_eval.py` | Added `conditional_concat_2stream` and the unconditioned `scoring_enhanced` to the ablation ladder; added the per-context reward `oracle`; new `profiles` / `pairs` / `hour_stride` / `include_oracle` args; restructured the loop so one `reset` per context serves every method and intent (12× fewer resets, which is what made six methods affordable); emits `hops_mean`, `n_paths_mean`, `optimality_gap` as trailing CSV columns and `ablation_per_context_rewards.npz`; `run_probing_ceiling` parameterized by intent; `run_intent_alignment` emits `loss_rate`, `hop_count`, `trust_own_w`. |
| `src/simulation/evaluation_env.py` | New public `evaluate_action()` (scores an action without advancing the clock — verified bit-identical to `apply_action`), `path_metrics_snapshot()`, `max_path_bandwidth()`. |
| `src/pipeline/chapter6_figures.py` | `metric="both"` two-panel ceiling with bin counts; corrected intent boxplots (+ loss-exposure panel); new `plot_probing_by_intent`, `plot_training_curves`, `plot_per_profile_training_curves`, `plot_intent_interpolation`, `plot_pathcount_scaling`. |
| `src/rl/path_selection_train.py`, `src/pipeline/dqn_train_cli.py` | `ScoringHyperparams.seed`, `DQN_SEED`, `--seed`, wired into all three trainers. |
| `src/rl/reward_profiles.py` | `probe_extreme` / `loss_only` profiles and the `conflicting` set (`DQN_CONDITIONAL_PROFILES=conflicting`). |
| `evaluation/05_evaluate_methods.py` | `.eval()` on every loaded agent; `epsilon=0` on the conditional agent; new `avg_inference_time_ms` (+ p50/p95/p99) timing only the decision; ECMP flow stub fixed to carry `source_as` / `destination_as` / `flow_id`. |
| `evaluation/eval_ablation_intent.py`, `eval_probing_ceiling.py`, `06b_generate_chapter6_figures.py` | New CLI flags for the above. |
| `tests/test_chapter6_eval.py` | Updated for the new method set; added an oracle-dominance assertion, a paired-array check, a permutation-equivariance test, and a seed-reproducibility test. |

**New**

`src/rl/seeding.py`, `evaluation/build_heldout_pairs.py`, `evaluation/eval_intent_interpolation.py`,
`evaluation/eval_pathcount_scaling.py`, `evaluation/analyze_pair_selection_spread.py`,
`evaluation/plot_training_curves.py`, `evaluation/analyze_seed_variance.py`,
`evaluation/run_seed_sweep.sh`.

**Newly trained checkpoints** (the previously missing unconditioned agents, plus the seed sweep):
`dqn_scoring_enhanced_model.pth`, `dqn_scoring_simple_model.pth`, `dqn_simple_model.pth`,
and `seeds/seed{1..5}/dqn_conditional_{scoring,value_concat,concat}_model.pth`. No existing checkpoint
was overwritten.
