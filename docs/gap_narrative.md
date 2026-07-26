# What still fits the Chapter 4 narrative

**Companion to:** `docs/gap_findings.md` (the raw results) and `docs/gap_experiments.md` (the work order).
**Written:** 2026-07-25. **Audience:** the agent writing `/home/octav/thesis-report`.

`gap_findings.md` reports what was measured, including results that cut against the draft. This document
answers the follow-up question: *given all of it, what story can Chapter 4 still tell without
contradicting its own data?* It is organised as a writing guide — what survives untouched, what needs a
narrower scope, where the new material slots in, and what must not be claimed.

---

## 0. The headline: the chapter's argument is already the defensible one

This is the single most important thing to know before rewriting anything, and it is easy to miss.

`sections/4_single_path.tex:236-240` already says:

> "the resulting $Q_i(s, \mathbf{w}, f_i)$ is not of the form assumed in \cref{prop:p1cancel}, the
> argument does not apply, and such a policy **can** re-rank. \Cref{sec:p1eval:ablation} measures one.
> Second, what distinguishes the design of the next section is therefore **not that it is the only
> mechanism that can re-rank**, but that it is the only one that re-ranks *by construction*."

The draft has already been rewritten off the indefensible claim. It explicitly (i) disclaims that FiLM
is the only workable mechanism, (ii) predicts that a policy reaching the per-path features can re-rank,
and (iii) promises that §4.5.2 measures such a policy. **The two-stream control does not break this
argument — it is the measurement the sentence already promises.** What was missing was the row in the
table, not a different thesis.

Concretely: the chapter's central claim is about **where** conditioning enters, not **how**. Everything
measured supports that claim, at higher confidence than before. The corrections needed are narrower than
`gap_findings.md` §1 might suggest in isolation, and none of them touch the chapter's spine.

---

## 1. Claims that survive unchanged — most are now *stronger*

Nothing in this section needs rewriting for correctness. Each item can additionally gain a confidence
interval, an upper bound, or a per-intent breakdown it did not have.

### 1.1 Proposition 4.1 and its empirical signature — now with confidence intervals

The draft's core empirical claim is that value-stream conditioning yields near-zero behavioural
divergence while conditioning that reaches the per-path features does not. Over **five training seeds**
(`gap/stats/seed_variance.json`):

| Method | Adaptivity, mean [95 % CI] |
|---|---|
| Value-Concat | 0.0425 [0.0309, 0.0541] |
| Conditional-FiLM | 0.1364 [0.1052, 0.1676] |
| Two-Stream-Concat | 0.1398 [0.1293, 0.1503] |

**The intervals for Value-Concat and both per-path variants are disjoint by a wide margin.** The
proposition's signature reproduces on every seed. This is now a statistical claim, not a single-run
observation, and it is the strongest result in the chapter.

The reward consequence holds too: on Low-Latency, FiLM 0.7308 [0.7302, 0.7313] vs Value-Concat
0.7058 [0.6996, 0.7119] — again disjoint. **The load-bearing 0.024 gap is real and reproducible.**

### 1.2 The "$2.5\times$ less re-ranking" claim — survives, and grows

`:790` says the value-stream variant "re-ranks $2.5\times$ less than FiLM while losing on the intent that
most demands re-ranking". Single seed: 0.1284 / 0.0502 = 2.56×. Over five seeds: 0.1364 / 0.0425 =
**3.2×**, with disjoint CIs.

Keep this sentence. Optionally strengthen it to "roughly threefold, with non-overlapping 95 % confidence
intervals over five training seeds". It is a FiLM-vs-Value-Concat statement, which is exactly the
comparison that survives.

### 1.3 The $40\times$ probing reduction — exact, and now per-intent

Contribution 3 and `:698`/`:710`/`:789` claim the selector "reaches the goodput of exhaustive
widest-path selection at roughly $40\times$ less probing… one probe per decision rather than thirty",
and "less than every other baseline in absolute terms". Measured:

* widest-path 5 388.1 ms/selection ÷ FiLM 135.1 ms = **39.9×** — "roughly 40×" is exact;
* **1 probe vs 30** — exact;
* FiLM 135.1 ms vs the cheapest heuristic's 359.7 ms — FiLM is **2.7× cheaper than the cheapest
  baseline**, so "less than every heuristic in absolute terms" is exact.

All three numbers stand. §3.3 below shows how to make the claim strictly stronger at no cost.

### 1.4 The single-path ceiling — entirely untouched

`sec:p1eval:ceiling` was not re-run and did not need to be. The two-panel figure
(`fig_ceiling_two_panel.png`) is generated **from the exact CSV the draft already cites**, so every
number in that prose is unchanged: FiLM's reward flat 0.896 → 0.886 while goodput falls 9 067 → 6 844
Mbps (−24.5 %); shortest-path 0.72 → 0.31; random 0.26 → −0.32; 1 792 decisions per bin.

The "why the reward does not see the ceiling" paragraph is also confirmed structurally: goodput is
normalized by the best candidate at the same hour (`evaluation_env.py:315-321`), so the reward measures
selection quality and is blind to the ceiling by construction. The chapter's bridge to Chapter 5 is
solid and now has the figure it asks for.

### 1.5 The permutation-equivariance claim — now measured, and exact

`sec:p1design:scoring` and Contribution 1 rest on the architecture's construction. Measured over
**32 256 permutation trials per agent**: the scoring and conditional agents select the *same underlying
path* in **100.0000 %** of trials. Not "approximately invariant" — exactly invariant.

### 1.6 Zero-shot generalization — now supported, so the claim can be restored

The `\gap{}` at `1_introduction.tex:61` asks for "zero-shot" to be restored once the experiment exists.
It exists. FiLM's chosen latency moves **monotonically** 17.31 → 12.84 ms across the intent
interpolation, Spearman ρ = **−1.000**, with no single step carrying more than 14.8 % of the span (a
memorized-endpoints policy would put ~100 % in one step). It reproduces on **pairs the agent never
trained on** (24.06 → 19.47 ms, ρ = −1.000).

**Restore "zero-shot"** in the abstract, Contribution 1, and `sec:analysis:generalization`, and write
§4.5.4 from `gap/zeroshot/`.

---

## 2. Claims that need a narrower scope — precision, not retraction

None of these are contradicted. Each is *true as measured but stated more broadly than the evidence
supports*. Narrowing them costs the chapter very little and removes every attackable sentence.

### 2.1 "A spectrum of objectives — throughput, latency, trust, probe-frugality"

**Where:** abstract; `1_introduction.tex:49`; `sec:p1eval:intent`.

**Problem.** Of the four evaluated intents, only two produce distinguishable behaviour, and the reason
is the environment rather than the mechanism:

* **98.85 % of chosen paths have exactly zero loss.** The Low-Loss intent has nothing to optimize; it
  achieves the same 0.48 % loss exposure as Throughput and Balanced.
* The latency-optimal and loss-optimal paths are the **same path 82.8 %** of the time.
* `w_probe` has ~5 % of the leverage of a real intent axis — even the *oracle* moves chosen hop count
  by 0.018 hops across a 350× change in `w_probe`.

**Fix.** Claim **one demonstrated trade-off axis, goodput ↔ latency**, and say why: a genuine
bandwidth/latency conflict exists in **59 %** of decision contexts, which is where conditioning pays.
Suggested replacement phrasing:

> "…a single policy then serves a range of application intents without retraining. In this environment
> the axis along which intents genuinely conflict is goodput against latency — loss is near-zero on
> 99 % of selections and probe cost is nearly path-independent — so intent alignment is demonstrated
> there, and the remaining profiles are served without being behaviourally distinguishable."

This is honest, still interesting, and pre-empts the obvious examiner question.

### 2.2 The $\approx$44 % goodput gap attributed to conditioning

**Where:** `:334`, `:525-526`.

**Problem.** The flat DQN reaches 5.65 Gbps against the conditional agents' 8.2 Gbps, and the draft
attributes the gap to conditioning. The architectural control now exists: the **unconditioned**
path-scoring DQN reaches **8 210 Mbps** with no intent input at all, and matches the conditional agents
on Throughput, Low-Loss and Balanced (it even edges FiLM on Balanced, 0.8978 vs 0.8939).

**Fix — and it makes the chapter's ladder cleaner, not weaker.** Split the attribution:

> "Per-path scoring accounts for the goodput jump: the unconditioned scoring agent already reaches
> \SI{8.21}{\giga\bit\per\second} against the flat agent's \SI{5.65}{\giga\bit\per\second}.
> Conditioning buys something different and narrower — the ability to re-rank when the intent conflicts
> with the bandwidth-maximizing default. On Low-Latency the conditioned policy scores 0.732 against the
> unconditioned scorer's 0.679, cutting chosen latency from \SI{17.0}{\milli\second} to
> \SI{12.8}{\milli\second}."

This is a *better* result for the thesis: it isolates each contribution to its own mechanism instead of
letting one absorb the other's credit, and it matches Contribution 1 and Contribution 2 being separate
contributions.

### 2.3 "Held-out pairs"

**Where:** `sec:p1eval:setup`, `:448`.

**Problem, corrected.** The draft's `\gap{}` assumes training used a 64-pair prefix. It did not:
episodes sample uniformly from all 8 010 pairs and visit **644 distinct pairs**; only **2 of the 32**
evaluation pairs were ever trained on. The evaluation set is weak because all 32 share **source AS 0**,
not because it was trained on.

**Fix.** Two sentences, and a genuinely held-out table now exists:

> "The main tables use a temporal split: 32 evaluation pairs, all sharing one source AS, over the 336
> held-out hours. Because episodes sample pairs uniformly from the full pool, only 2 of these 32 were
> ever visited in training. To test structural generalization we additionally construct an exactly
> disjoint set of 32 pairs spanning 32 distinct source ASes, none visited during training
> (\cref{tab:heldout})."

The result to report is favourable: the conditional agents' optimality gap stays tiny (0.0014 in-pool →
0.0037 held-out) and adaptivity *rises* (0.128 → 0.151), while the **flat DQN's Throughput reward goes
negative** (0.333 → −0.085). That last number is strong support for abandoning the fixed-action
architecture, which is what `sec:p1design:strawman` argues.

### 2.4 "Matching the heuristic that is strongest for it"

**Where:** chapter summary, `:787`.

This is true, and was previously only measured under Balanced. It is now measured per intent and the
claim can be made much more forcefully — see §3.3.

---

## 3. New material, and exactly where it slots in

| Thesis location | What to add | Source |
|---|---|---|
| `sec:p1design:env` (after the link/path model) | Environment-realism paragraph: best candidate carries **3.04×** the median candidate's goodput (p90 18.3×), best−worst **7 983 Mbps**, best path changes hour-to-hour in **8.6 %** of hours. Settles §4.1.1's assertion with a number. | `gap/realism/` |
| `sec:p1impl:agents` (agent ladder) | The unconditioned enhanced scoring agent now exists and is trained; add it to the list and the table. | `dqn_scoring_enhanced_model.pth` |
| `tab:p1eval:ablation` | Two new rows (**Scoring DQN (uncond.)**, **Two-Stream-Concat**) and a bottom **Oracle** row turning the table into a normalized optimality gap. Exact `Params`: flat 85.3 k, uncond. scoring 36.2 k, Value-Concat 36.9 k, Two-Stream 37.5 k, FiLM 38.8 k. | `gap/ablation_main/table_6_1.tex` |
| `sec:p1eval:ablation` (new paragraph) | Seed variance: means with 95 % CIs over 5 seeds, plus the paired Wilcoxon over 10 752 shared contexts. | `gap/stats/` |
| **`sec:p1eval:zeroshot`** (currently empty) | Write it from the interpolation sweep — in-pool and held-out. | `gap/zeroshot*/` |
| **`sec:p1eval:dynamic`** (currently empty) | Write it from the path-count subsampling + order invariance. | `gap/pathcount/` |
| `sec:p1eval:intent` | Corrected boxplot figure; the 98.85 %-zero-loss explanation. | `gap/intent/` |
| `sec:p1eval:overhead` | Per-intent probing table (§3.3). | `gap/probing/` |
| `fig:p1eval:ceiling` | Replace with the two-panel goodput+reward version. | `fig_ceiling_two_panel.png` |
| `sec:analysis:deployability` (`6_synthesis.tex:103`) | Inference latency **0.48 ms** mean, **2.2 ms** p99, CPU-only, scoring 30 paths. | `evaluation_results.json` |
| `app:extra` | Training curves + per-intent curves. | `gap/training/` |
| `app:p1hyper` | Seeds column = **5**; wall-clock ≈1–3 min per agent. | — |

### 3.1 The oracle turns Table 4.1 into an optimality gap

A per-context reward `argmax` under the intent being scored, sharing the agents' contexts and
normalization. FiLM's shortfall: **0.19 %** (Low-Latency), 0.22 % (Low-Loss), 0.45 % (Balanced),
0.60 % (Throughput).

This lets the chapter **restore "near-oracle"**, which was removed pending exactly this — now as a
measured statement: *"within 0.6 % of a per-context reward oracle on every evaluated intent."*

One caveat worth a sentence: the oracle's own adaptivity is **0.1609**, so 0.16 — not 1.0 — is the
ceiling for the Adaptivity metric in this environment. FiLM at 0.1284 attains **80 %** of the achievable
re-ranking. This reframes numbers that otherwise look small on a $[0,1]$ scale, and it is a point in the
chapter's favour.

### 3.2 §4.5.4 Zero-Shot Generalization — ready to write

Sweep $\mathbf{w}(t) = (1-t)\cdot$Throughput $+\ t\cdot$Low-Latency, $t \in \{-0.2, \dots, 1.2\}$; only
$t = 0, 1$ were trained on.

| Method | latency $t{=}0 \to t{=}1$ | monotone | ρ | largest step |
|---|---|---|---|---|
| Conditional-FiLM | 17.31 → 12.84 ms | yes | −1.000 | 14.8 % |
| Two-Stream-Concat | 18.40 → 12.73 ms | yes | −1.000 | 24.7 % |
| Value-Concat | 17.07 → 15.11 ms | yes | −1.000 | 17.1 % |

Note the secondary finding, which supports Proposition 4.1 from a new direction: Value-Concat
interpolates smoothly but traverses only **2.0 ms** of behaviour against FiLM's 4.5 ms — it can move
behaviour only through indirect training effects, so it covers less of the space.

### 3.3 §4.5.6 Probing Overhead — a strictly stronger claim, free

Previously run only under Balanced. Per intent, same 10 752 contexts:

| | probes/sel | ms/sel | Throughput | Low-Latency | Low-Loss | Balanced |
|---|---|---|---|---|---|---|
| **Conditional-FiLM** | **1** | **132–137** | 0.980 | 0.732 | **0.939** | **0.894** |
| Widest-Path | 30 | 5 388 | **0.982** | 0.652 | 0.930 | 0.891 |
| Lowest-Latency | 30 | 360 | 0.449 | **0.733** | 0.899 | 0.629 |

> One conditioned policy tracks the per-intent strongest heuristic — **and which heuristic is strongest
> changes with the intent**. Widest-path wins under Throughput by 0.002; lowest-latency wins under
> Low-Latency by 0.001; FiLM beats every heuristic under Low-Loss and Balanced. Each heuristic collapses
> under the other's objective (widest-path 0.652 under Low-Latency; lowest-latency 0.449 under
> Throughput). The learned selector does this at 1 probe instead of 30.

That is the claim `sec:disc:interpretation` already asserts, now measured.

---

## 3.4 Decision taken 2026-07-26: FiLM was dropped from the thesis

**Status: resolved.** The thesis now ships **Two-Stream-Concat** as its conditioning mechanism, and FiLM
has been removed from every chapter. The reasoning, recorded so it is not re-litigated:

* **Chapter 5 uses neither FiLM nor intent conditioning** — the words "FiLM" and "intent" appear zero
  times in `sections/5_multipath.tex`. The only thing the two chapters share is the
  permutation-equivariant scoring architecture, so FiLM was carrying complexity for one chapter only.
* **The five-seed data does not support keeping it.** §2.11 finds the two statistically
  indistinguishable on all four intents and on adaptivity, while FiLM is 1 294 parameters larger and has
  3× the seed-to-seed spread. §4 items 1–3 of this document already ruled out every empirical argument
  for FiLM; what was left was a design-preference argument, which did not justify the exposition cost.
* **What survives is Proposition 4.1**, which is a claim about *where* conditioning enters and is
  mechanism-agnostic. It is now the chapter's load-bearing result.

**Work done to re-anchor the data.** `run_intent_alignment` and `run_probing_ceiling` in
`src/pipeline/chapter6_eval.py` hard-coded `conditional_film`; both now take an `agent_key` argument
(`--agent` on the two CLI scripts, default unchanged). Re-run with
`--agent conditional_concat_2stream` into `run_20260722_180329/gap/shipped_2stream/`; the published
`gap/` artifacts were **not** overwritten. `plot_training_curves.py` gained `--exclude LABEL`. The
zero-shot and path-count CSVs were filtered rather than re-evaluated, since they already carry every
method.

**Every headline claim held or strengthened** under the new agent:

| Claim | FiLM | Two-Stream |
|---|---|---|
| Probing reduction vs widest-path | 39.9× | 39.4× |
| Goodput vs widest-path | −1.14 % | **−0.44 %** |
| Ceiling drop, light→heavy | 24.5 % (9067→6844) | 24.4 % (9098→6874) |
| Reward across congestion | 0.896→0.886 | 0.898→0.889 |
| Beats *every* heuristic on | 2 of 4 intents | **3 of 4** |
| Max optimality gap | 0.60 % | **0.25 %** |
| Heatmap diagonal wins | 2 of 4 columns | **4 of 4** |
| Inference latency (mean) | 0.40–0.48 ms | **0.20–0.29 ms** |

Two numbers genuinely changed and are reported as measured: loss exposure under Low-Latency is 4.91 %
(was 3.15 %), and the intent heatmap's diagonal now wins every column, so the old "the diagonal does not
win on two columns" explanation no longer applies and was replaced by a column-range discussion.

---

## 4. What must not be claimed

Short list of sentences the data will not support. None of these appear in the current draft as far as I
can tell — this is a guard rail for the rewrite.

1. **"FiLM outperforms per-path concatenation."** Over five seeds the two are statistically
   indistinguishable on all four intents and on adaptivity (FiLM [0.1052, 0.1676] vs Two-Stream
   [0.1293, 0.1503] — heavily overlapping).
2. **"FiLM is the cheaper / more parsimonious mechanism."** It is **1 294 parameters larger** than
   Two-Stream-Concat (38.8 k vs 37.5 k). The parsimony argument runs backwards. The available argument
   is the one the draft already makes at `:238-240`: FiLM re-ranks *by construction*, and its
   conditioning surface is explicit and independent of the number of paths — a design preference, stated
   as such, not an empirical win.
3. **"FiLM is the more reliable mechanism."** Its adaptivity standard deviation across seeds is 0.0251
   against Two-Stream's 0.0084 — a 3× wider spread.
4. **"Loss-aversion and probe-frugality are demonstrated intent axes."** See §2.1.
5. **"Better-chosen intent profiles would fix the degeneracy."** Tested directly: a `probe_extreme`
   profile at 16× the probe weight moves the *oracle's* hop count by 0.009 hops, and a `loss_only`
   profile collapses onto pure throughput. Since the oracle bounds every policy, no profile choice can
   help — the constraint is the reward/environment. State it as Future Work: add an explicit hop-count
   term, or raise offered load until loss is common (only 7.8 % of links are oversubscribed today).
6. **"ECMP performs load balancing here."** It is evaluated with a constant flow identifier, so it pins
   every flow of a pair to one path. Label it *hash-pinned shortest path*, as the `\gap{}` at `:416`
   says.
7. **Proposition 4.1 holds exactly in the deployed system.** It holds for the *architecture*. In the
   environment, per-path feature 6 is `trust`, computed from $w_3, w_4$
   (`evaluation_env.py:397`), so intent does leak into the path features — which is why Value-Concat's
   adaptivity is 0.050 rather than 0. Worth one honest sentence; it makes the proposition sharper, not
   weaker, because the measured residual is exactly what the leak predicts.

---

## 5. The arc that still works, end to end

Stated as the chapter would tell it, with every claim backed:

1. **A fixed-objective, fixed-action DQN is the wrong shape for this problem.** It cannot handle a
   variable candidate set, and it does not generalize: its Throughput reward goes *negative* on pairs it
   never trained on. Replacing it with a permutation-equivariant per-path scorer buys the bulk of the
   performance — 5.65 → 8.21 Gbps — and gives exact order invariance (100.0000 % over 32 256 trials) and
   flat regret from $N = 2$ to $N = 30$.
2. **Making one policy serve many intents is not a matter of feeding it the intent.** Proposition 4.1
   says conditioning that enters only path-independent terms cancels under the $\arg\max$; measurement
   confirms it across five seeds with disjoint confidence intervals (adaptivity 0.042 vs 0.136). The
   design lesson is about *where* the intent enters. Which per-path mechanism is used — multiplicative
   modulation or concatenation — is not what decides the outcome; FiLM is chosen because its effect on
   the ordering is structural rather than learned.
3. **The conditioned policy generalizes over the intent space, not just across trained points.**
   Behaviour interpolates monotonically between trained intents (ρ = −1.000) and does so on pairs never
   seen in training.
4. **It does this at near-oracle quality and a fraction of the measurement cost.** Within 0.6 % of a
   per-context reward oracle on every intent; tracks whichever heuristic is strongest for the current
   intent while issuing 1 probe instead of 30, and costs less than every heuristic in absolute terms.
5. **And it still hits a wall.** Even optimal single-path selection loses ~25 % of its goodput from
   light to heavy congestion, while its *reward* stays flat — the policy is doing its job and the action
   space is the limit. That is what Chapter 5 sets out to break.

Steps 1–4 are all now measured. Step 5 is unchanged. The one honest scoping sentence the chapter needs
is that this environment offers a single genuine trade-off axis (goodput ↔ latency), so intent alignment
is demonstrated there rather than across all four evaluated profiles.

---

## 6. Priority order for the rewrite

1. **Write `sec:p1eval:zeroshot`** — the chapter's most conspicuous hole, and the result is positive.
2. **Add the two rows + oracle to Table 4.1** and the seed CIs. Restores "near-oracle", closes the
   two-stream gap, and makes the ablation a normalized optimality gap.
3. **Split the 44 % attribution** (§2.2) — cheap, and it strengthens the ladder.
4. **Narrow the "spectrum of objectives" language** (§2.1) — removes the most attackable sentence.
5. **Write `sec:p1eval:dynamic`** from the path-count study.
6. **Swap in the corrected boxplots and two-panel ceiling figure.**
7. Per-intent probing table; held-out-pairs paragraph; environment-realism sentence; appendix training
   curves.
