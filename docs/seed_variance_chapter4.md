# Chapter 4 over five training seeds

Run: `evaluation/run_20260722_180329`, seeds 1–5, agent **Two-Stream-Concat**. Aggregates in
`seeds/aggregate/`, figures in `seeds/aggregate/figures/` (installed into `~/thesis-report/figures/`).

**Headline: five of the eight sharp claims in the chapter do not survive five seeds.** None of the
five is a collapse — the mechanism is intact everywhere and most of the reversals are third-decimal —
but each is stated in the chapter at a precision the spread does not support, and three of them are
stated as exact.

## Why

Chapter 4 reported its conditioning ablation (`tab:p1eval:seeds`) over five training seeds with
confidence intervals, and every *other* quantitative result in the chapter from a single reference
training run. Chapter 5 of the same thesis makes single-run reporting its central methodological
cautionary tale — four results there did not survive being run three times, and it argues in print
that the spread over *training* seeds exceeds the confidence interval over *evaluation* seeds by an
order of magnitude on this class of problem. Applying that lesson to only one of the two systems is
the asymmetry an examiner finds first, and these runs cost minutes.

## Protocol

Unchanged from `tab:p1eval:seeds`, and enforced rather than assumed:

* **Only the learning process varies.** `run_seed_sweep.sh` reseeds torch/numpy/random; the
  environment's pair, hour and profile RNGs stay fixed, so every seed sees an identical stream of
  training contexts.
* **Identical evaluation contexts.** `run_seed_result_sweep.py` loads the run context once and hands
  the same object to every study of every seed, so all five are graded on the same 10752 held-out
  decision contexts (32 pairs × 336 hours, the last 14 days).
* **No FiLM.** The variant was dropped from the thesis on 2026-07-26. Its checkpoints are still on
  disk in the seed directories, so each seed is evaluated through a staging directory
  (`seeds/seed<N>/shipped/`) that links only the four reported rungs. `run_seed_sweep.sh` no longer
  trains it.
* **Intervals** are the t-based 95% CI over seeds from `analyze_seed_variance._ci95`, reused directly
  so every seeded number in the chapter is computed one way.

## Claim survival

| # | Claim (as the chapter states it) | Five-seed verdict |
|---|---|---|
| 1 | Intent-alignment diagonal wins **all four** columns | **FAILS** — wins every seed on 2/4 |
| 2 | Zero-shot Spearman is **exactly** ρ = −1.000 | **FAILS** — one seed at −0.9909 |
| 3 | No single interpolation step carries more than **24.7%** of the span | **FAILS** — worst seed 31.0% |
| 4 | Order invariance is **exactly** 100.0000% | **SURVIVES** — 483840 trials, zero mismatches |
| 5 | Beats every heuristic on **three of four** intents | **FAILS** — 2/4 in every seed, 3/4 on the mean |
| 6 | Within **0.0015** of lowest-latency on Low-Latency | **FAILS** — 0.0031 [0.0015, 0.0048] behind |
| 7 | Ceiling descends **~24%** (9.1 → 6.9 Gbit/s) | **SURVIVES** — 24.59% [24.28, 24.90] |
| 8 | Reward stays flat (0.898 → 0.889) | **SURVIVES** — 0.8976 → 0.8883 |

Two further numbers the chapter quotes in passing also move and are listed with the results below:
goodput "within 0.5% of widest-path" is now 0.74% [0.48, 1.00] behind, and the probing reduction
"roughly 39×" is 39.6× [39.5, 39.8] and holds.

---

## 1. Intent-alignment matrix (`tab:p1eval:heatmap`, `fig:p1eval:heatmap`) — **claim fails**

| Column (objective) | Diagonal wins | Margin over best off-diagonal | Column range | Nearest rival |
|---|---|---|---|---|
| Throughput  | **5/5** | +0.0121 [+0.0091, +0.0152] | 0.244 | Balanced |
| Low-Latency | **5/5** | +0.0197 [+0.0146, +0.0248] | 0.064 | Low-Loss |
| Low-Loss    | 4/5 | +0.0011 [−0.0018, +0.0039] | 0.011 | Balanced |
| Balanced    | 2/5 | **−0.0005** [−0.0031, +0.0021] | 0.113 | Throughput |

The two columns the chapter already flagged as weak are the two that break, and the **Balanced column
reverses on the mean**: being told *Throughput* scores 0.897 under the Balanced objective against
0.896 for being told Balanced. The Low-Loss diagonal still leads on the mean but by 0.0011 against a
CI half-width of 0.0029, so it is not distinguishable from its rival.

This is not a failure of intent alignment — it is a sharper version of what the chapter already says.
Alignment is real and large on the two objectives that conflict with the bandwidth-maximizing default
(margins ~4× their own confidence half-widths) and is indistinguishable from noise on the two that
do not. `fig:p1eval:heatmap` now marks the two non-significant diagonals with a dashed box and an
`n.s.` tag, and prints each column's range in its axis label, so the figure cannot be read as showing
four wins.

**Wording that would hold:** "the diagonal wins the two columns that demand re-ranking, with margins
far outside the seed spread; on Low-Loss and Balanced the four intents are statistically
indistinguishable, which is what a column range of 0.011 predicts."

## 2. Per-intent chosen-path metrics (`fig:p1eval:boxplots`) — **mechanism survives, one number moves a lot**

| Intent | Chosen latency (ms) | Chosen goodput (Mbps) | Trust | Selections with loss > 0 |
|---|---|---|---|---|
| Throughput  | 17.79 [17.26, 18.32] | 8232 [8221, 8243] | 0.9110 [0.9083, 0.9136] | 0.484% (exact, all seeds) |
| Low-Latency | 12.89 [12.33, 13.44] | 7296 [7015, 7577] | **0.9333** [0.9328, 0.9339] | 4.51% [1.42, 7.60] |
| Low-Loss    | 14.85 [14.23, 15.46] | 7868 [7532, 8203] | 0.9257 [0.9226, 0.9288] | 0.755% [0.392, 1.118] |
| Balanced    | 16.56 [16.23, 16.89] | 8181 [8160, 8203] | 0.9171 [0.9155, 0.9188] | 0.484% (exact, all seeds) |

Everything qualitative holds. Low-Latency saves **4.90 ms [4.16, 5.64]** against Throughput (chapter:
5.7 ms from 18.40 → 12.73; the seed mean is 17.79 → 12.89). The counter-intuitive trust result holds
and is significant: trust(Low-Latency) − trust(Low-Loss) = **+0.0077 [+0.0041, +0.0112]**.

The number to restate is **loss exposure under Low-Latency**. The chapter reports 4.91%; the seed mean
is 4.51% but the per-seed values are 2.21, 3.35, 3.44, 4.93, 8.61 — a 3.9× spread, by far the least
stable quantity in the chapter. The direction (Low-Latency is the only intent that meaningfully raises
loss exposure) holds in every seed; the magnitude should be given as a range, not as 4.91%.

Two smaller corrections: Low-Loss no longer sits exactly on the 0.484% floor (0.755% [0.392, 1.118]),
so "the 0.48% floor that the other three intents all sit on" is true of Throughput and Balanced only;
and the share of chosen paths with exactly zero loss is 98.44% over five seeds against the 98.85%
quoted.

## 3. Zero-shot interpolation (`fig:p1eval:zeroshot`) — **both sharp claims fail, the result holds**

| | Two-Stream-Concat | Value-Concat |
|---|---|---|
| Latency t=0 → t=1 | 17.79 → 12.89 ms | 16.86 → 15.29 ms |
| Span | 4.90 ms [4.16, 5.64] | 1.56 ms [1.22, 1.91] |
| Extrapolation at t=1.2 | 11.98 ms | 14.68 ms |
| Spearman ρ per seed | −1.000, −1.000, −1.000, **−0.9909**, −1.000 | −1.000 ×5 |
| Largest single step | 23.3, 23.2, 23.2, 20.4, **31.0%** | 12.4–20.8% |

**ρ = −1.000 is not exact across seeds.** Seed 4 has one non-monotone step, between t = 0.0 and
t = 0.1, of **+0.036 ms** — 0.6% of that seed's 5.47 ms span, and every other step descends. Four of
five seeds are exactly monotone.

**The 24.7% claim fails.** Seed 5 concentrates 31.0% of its span in one step; the mean is 24.2%
[19.3, 29.2]. The claim's *purpose* still holds — a memorizing policy would put essentially the whole
span in one step, and the worst seed puts under a third of it there — but the specific threshold does
not survive.

The Value-Concat contrast strengthens: Two-Stream traverses **3.13×** the behavioral span, against the
chapter's "less than half as much" (5.7 vs 2.0 ms). `fig:p1eval:zeroshot` now shows all five per-seed
traces behind the mean and its band, so the monotonicity claim can be checked per run.

**Wording that would hold:** "ρ = −1.000 in four of five seeds and −0.991 in the fifth, whose single
ascending step is 0.036 ms; no seed concentrates more than 31% of the span in one step."

## 4. Path-count scaling and order invariance (`fig:p1eval:pathcount`) — **survives, including the exactness claim**

Order invariance is **exactly 100.0000%** for every scoring and conditional agent in every seed:
161280 permutation trials per agent (32256 per seed × 5), 483840 in total, **zero** mismatches. This is
the one exactness claim in the chapter that is exact, as it should be — it follows from the
architecture rather than from training. Note the agent count changes with FiLM dropped: the claim now
covers **three** scoring and conditional agents, not four.

Mean regret, seed means with 95% CI:

| N | Flat DQN | Scoring DQN | Value-Concat | Two-Stream-Concat |
|---|---|---|---|---|
| 2  | 0.2385 [0.2261, 0.2510] | 0.0063 | 0.0046 | 0.0064 |
| 4  | 0.4325 [0.4041, 0.4608] | 0.0094 | 0.0087 | 0.0109 |
| 8  | 0.5163 [0.4802, 0.5523] | 0.0088 | 0.0093 | 0.0110 |
| 16 | 0.4904 [0.4586, 0.5222] | 0.0048 | 0.0075 | 0.0063 |
| 30 | 0.3212 [0.3119, 0.3306] | 0.0003 | 0.0056 | 0.0016 |

The chapter's "0.0001 to 0.013" for the scoring agents becomes **0.0003 to 0.0110**, and its "0.24 to
0.50" for the flat DQN becomes **0.239 to 0.516** — both hold. The two-orders-of-magnitude separation
holds with disjoint intervals at every N. The caveat about being better at N = 30 than at small N
holds for the unconditioned scorer and Two-Stream but not for Value-Concat, whose regret is flat
across N.

## 5. Probing overhead (`tab:p1eval:probing`, `fig:p1eval:probing`) — **claim fails on 1 of 4 intents**

Mean reward by method and intent, five-seed means (± is the CI half-width; heuristics are
deterministic and have none):

| Method | Probes | ms/sel. | Throughput | Low-Latency | Low-Loss | Balanced |
|---|---|---|---|---|---|---|
| **Two-Stream-Concat** | **1** | **136.0** [135.4, 136.5] | 0.9839 ±0.0025 | 0.7296 ±0.0016 | **0.9393** ±0.0025 | **0.8963** ±0.0013 |
| Widest-Path    | 30 | 5388 | **0.9820** | 0.6521 | 0.9295 | 0.8913 |
| Lowest-Latency | 30 | 360  | 0.4489 | **0.7327** | 0.8991 | 0.6294 |
| ECMP           | 30 | 360  | 0.5770 | 0.7109 | 0.9117 | 0.6930 |
| SCION-Default  | 30 | 360  | 0.4166 | 0.7258 | 0.8931 | 0.6098 |
| Shortest-Path  | 30 | 360  | 0.3831 | 0.7212 | 0.8883 | 0.5904 |
| Random         | 30 | 360  | −0.5690 | 0.2475 | 0.5578 | −0.0882 |

* **Beats every heuristic on 2 of 4 intents in every seed** (Low-Loss, Balanced), not three. On
  Throughput the learned selector leads on the mean (0.9839 vs 0.9820) but loses to widest-path in
  **1 of 5 seeds**, and the margin's interval [−0.0006, +0.0044] includes zero.
* **The 0.0015 gap on Low-Latency does not hold.** The shortfall behind lowest-latency is
  **0.0031 [0.0015, 0.0048]** — the reference run's 0.0015 is the *best* case across seeds, not the
  typical one. The selector is still ahead of every other heuristic on that intent, including
  SCION-Default (0.7258) and shortest-path (0.7212).
* **Probing reduction survives:** 39.6× [39.5, 39.8] against widest-path, and the learned selector is
  still the cheapest method in the comparison (136.0 ms vs 360 ms for the load-blind heuristics).
* **"Within 0.5% of widest-path" on goodput does not.** It is 0.74% [0.48, 1.00] behind
  (8181 vs 8242 Mbps).

**Wording that would hold:** "beats every heuristic on two of four intents in every seed and on three
of four on the mean; on Low-Latency it is within 0.003 of the heuristic purpose-built for that
objective, while beating every other one."

## 6. Single-path ceiling (`fig:p1eval:ceiling`) — **survives cleanly**

| Method | Goodput, light → heavy | Drop | Reward, light → heavy |
|---|---|---|---|
| Two-Stream-Concat | 9.10 → 6.86 Gbit/s | **24.59% [24.28, 24.90]** | 0.8976 [0.8958, 0.8993] → 0.8883 [0.8870, 0.8896] |
| Widest-Path       | 9.21 → 6.88 Gbit/s | 25.3% | 0.888 → 0.885 |
| Shortest-Path     | 7.37 → 3.11 Gbit/s | 57.8% | 0.724 → 0.312 |
| Random            | 4.19 → 0.55 Gbit/s | 87.0% | 0.260 → −0.325 |

The chapter's two quoted endpoints (9.1 → 6.9 Gbit/s) reproduce to one decimal, the ~24% descent
reproduces with a CI 0.6 percentage points wide, and the flat-reward claim reproduces (0.898 → 0.888
against the quoted 0.898 → 0.889). This is the chapter's structural bridge and it is the most robust
result in it — unsurprising, since the ceiling is a property of the environment that the policy tracks
rather than a property of the policy.

---

## What the numbers say overall

The five failures divide cleanly into two kinds, and the distinction is worth making in the chapter:

1. **Claims stated as exact that are not** (the ρ = −1.000, the 24.7% step, the four-column diagonal).
   In each case the underlying phenomenon survives with room to spare and only the exactness fails.
2. **Claims read off the best seed** (the 0.0015 Low-Latency gap, "three of four intents", "within
   0.5% of widest-path"). The reference run is at or near the favourable end of the seed distribution
   on each. This is exactly the failure mode Chapter 5 §5.4.9 documents, now measured on Chapter 4.

Nothing here touches the chapter's four summary claims: the architectural gap (flat vs per-path
scoring), Proposition 4.1's re-ranking prediction, zero-shot interpolation as a qualitative property,
and the ceiling all survive. What does not survive is the third decimal place.

## Parameter counts (`tab:p1eval:ablation`) — **confirmed correct**

Computed from `q_network` state dicts; identical in all five seeds.

| Method | Parameters | Table |
|---|---|---|
| Flat DQN | 85,343 | 85.3 k ✓ |
| Scoring DQN (uncond.) | 36,226 | 36.2 k ✓ |
| Value-Concat | 36,866 | 36.9 k ✓ |
| Two-Stream-Concat | 37,506 | 37.5 k ✓ |

The deployability sentence is safe: 37.5 k against 85.3 k is 44% of the size, i.e. "less than half".

## Reproducing

```bash
cd evaluation
./run_seed_sweep.sh run_20260722_180329                       # checkpoints + ablation (already on disk)
uv run python run_seed_result_sweep.py run_20260722_180329    # the other five results, per seed (~50 min)
uv run python analyze_seed_results.py run_20260722_180329     # aggregate + claim survival
uv run python plot_seed_figures.py run_20260722_180329 --copy-to ~/thesis-report/figures
```
