# Evidence ledger — what the context axis has and has not been shown to do

Written 2026-08-31. This file exists because four different claims have been
travelling under one label ("the ctx axis"), and mixing them has produced both
overstatement and understatement of what we measured. Each claim below is
stated separately, with its own measurement, its own verdict, and its own
unresolved parts.

Nothing here is new measurement. It is a reconciliation of results already in
this repository and in the vLLM threads, pinned to the commits that hold them.

---

## The four claims

| | Claim | Regime measured | Verdict |
|---|---|---|---|
| **A** | Enabling DSD costs a baseline tax before it drafts anything | ctx 400–38 000, c 2–189 | **Established.** ctx-flat, batch-scaling. Two independent external reproductions |
| **B** | Speculation's own benefit rises with context | ctx 400–4 000, c 256 | **Established.** K3/K0 crosses 1.0 near ctx 1 900 |
| **C** | A (batch × ctx) schedule beats a batch-only schedule | ctx 400–4 000, c 256 | **Met its pre-registered criterion against `A′` (1.29× / 1.30× / 1.36×); no advantage established over `static K=3`.** See §C |
| **D** | K\* shifts enough with context that mis-picking K is expensive | ctx 4 000 & 32 768, b 11 | **WEAK → CONCEDE.** Direction reproduced twice, magnitude +3.60 % (1.12σ) below the 2σ bar |

**C and D are not the same claim, and conflating them is the error this file
fixes.** C asks whether routing K by context beats routing it by batch alone,
and was measured under batch pressure (c = 256). D asks how sharp the K optimum
is at a fixed low batch (b = 11), and was measured without batch pressure. C
passed its pre-registered bar against `A′`; D did not pass its own. Neither
result invalidates the other. C's comparator was weak, and the stronger control
measured in the same run is covered in §C.

---

## A — The DSD baseline tax

Source: [`ctx_tax_mechanism/RESULTS.md` @ `a4bc8e5`](ctx_tax_mechanism/RESULTS.md).
1× H100 NVL (93.6 GB), vLLM 0.27.1, V1, target `prithivMLmods/gemma-4-31B-it-qat-FP8`,
drafter `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`. The speculative
arm is a K≡0 schedule; `spec_decode_num_draft_tokens_delta = 0.0` over every
measured run, so this is the cost of enabling the feature, not the cost of using it.

| ctx | concurrency | `no_spec` TPOT | `dsd` K≡0 TPOT | tax |
|---:|---:|---:|---:|---:|
| 400 | 2 | 14.374 ms | 15.317 ms | +6.56 % |
| 38 000 | 2 | 15.945 ms | 16.975 ms | +6.46 % |
| 400 | 189 | 77.164 ms | 93.820 ms | +21.58 % |

Context axis at fixed batch: −0.10 pp across a 95× range. Batch axis at fixed
context: +15.02 pp across a 90× range.

**Decomposition.** Two named terms, plus two found later:

- *graph term* — enabling dynamic SD forces `cudagraph_mode` from
  `FULL_AND_PIECEWISE` down to `PIECEWISE`; the engine logs the downgrade itself.
- *pool term* — the drafter costs ≈15 025 tokens of KV pool (10.6 %) while never
  drafting. The memory profiler attributes +3.93 GiB of a 4.97 GiB loss to the
  activation peak reserved for the drafter's forward and +1.05 GiB to weights,
  so this is a capacity reservation, not a draft KV allocation.
- *sync-forward term* — the K=0 early return sits after the drafter forward, so a
  forward runs every step. Bounded at +0.71 % TPOT on this stack; measured at
  4–11 % on other drafter architectures (vLLM #53420, #53426).
- *prefix-cache term* — the arms do not share a hit rate. Constant deficit of
  512–1 024 tokens, ctx-independent, so its share collapses as prompts grow
  (−12.89 pp of hit rate at ctx 400 against −0.09 pp at ctx 38 000).

**External reproductions.** Two, on stacks we do not control:

- vLLM #49986 / #49548 (Suppressor72) — same tax, different drafter architecture.
- vLLM #48494 (stefanskiasan, 2026-08-30) — 8× MI350X / ROCm / `GLM-5.3-Flash`
  FP8 MoE. Static-K + `FULL_AND_PIECEWISE` against DSD + `PIECEWISE`: −48 % at
  c=1, −32 % at c=8, −18 % at c=96. Their conclusion: "the documented fallback is
  a net regression."

**Not yet attributed.** The +15.02 pp of batch scaling is still bundled across
the terms above. The 2×2 attribution design that splits them is written and
pre-registered ([`tax_attribution/PREREGISTRATION.md` @ `286bcb8`](tax_attribution/PREREGISTRATION.md))
but **has not been run**.

**Caveat on the external comparison.** stefanskiasan's delta shrinks with
concurrency where ours grows. These are not the same measurement: their DSD arm
changes graph mode *and* re-picks K (at c=96 it picks K=1, worth +9 % over K=4 by
their own table), while ours holds K≡0. The two are consistent with a graph term
that dominates at low concurrency and a capacity term that dominates when
pool-limited, but that reading is unconfirmed until the 2×2 runs.

---

## B — Speculation's benefit against context

Source: Table 1 of the [PR #48944 decomposition comment](https://github.com/vllm-project/vllm/pull/48944#issuecomment-5091663057)
(position-balanced 2 trials, autotune cache wiped per launch, cold-start burn,
3 warmup discarded + 3 measured per cell, c = 256, per-arm stdev 0.01–1.72 % across arms).

| ctx | `no_spec` (K=0) | static-K3 | K3/K0 |
|---:|---:|---:|:---:|
| 400 | 2 711.7 | 2 139.8 | 0.79× |
| 900 | 1 987.1 | 1 838.9 | 0.93× |
| 1 900 | 1 815.2 | 1 822.4 | 1.00× |
| 4 000 | 1 535.9 | 1 692.8 | 1.10× |

Speculation *loses* at short context on this stack and crosses break-even near
ctx 1 900. The mechanism is visible in TPOT: as ctx doubles from 970 to 1 990 in
the earlier sweep, K=3 TPOT stays flat while K=0 TPOT keeps climbing —
long-context decode is bound by the per-step KV read, and verifying K drafted
tokens amortizes that read across K+1 tokens.

**Superseded numbers.** An earlier section of [`RESULTS.md`](RESULTS.md) reported
1.17× / 1.25× / 1.38× / 1.36× for this effect. Those carry warm-up bias
(single-run cells, no per-launch cache wipe, APC accumulation drifting one cell
1.28 → 1.32 → 1.36). They are retained in that file as the original observation
and are **marked superseded there**; the table above is the rigorous version.
Do not cite the 1.38× figure.

**Bound.** The gain is not unbounded. It peaks near ctx ≈ 2k and recedes at 4k as
K=3's own per-step cost begins rising. That rise was not decomposed.

---

## C — Context-routed K against batch-only K

Same source and methodology as B. `A′` is a 3-item batch-only schedule; `C′` is
the 6-cell 2-D schedule. This is the contrast the pre-registration named as
primary, with rules frozen 2026-07-27 before measurement (SUCCESS = ≥3/4 cells at
1.15×, one ≥1.30×).

**[Correction, 2026-10-22]** This section previously showed only the `C′` and
`A′` columns. The same campaign measured two further arms; all four are below.
The left column is the **configured** `--prefix-repetition-prefix-len`. Observed
per-request input lengths were 496 / 996 / 1 996 / 4 096, output length 100.

| configured prefix | `no_spec` | `static K=3` | `A′` (batch-only) | `C′` (2-D) | **C′/A′** | **C′ vs static** |
|---:|---:|---:|---:|---:|:---:|:---:|
| 400 | 2 711.7 | **2 139.8** | 1 875.6 | 1 890.7 | 1.01× | **−11.64 %** |
| 900 | 1 987.1 | 1 838.9 | 1 453.5 | 1 874.6 | **1.29×** | +1.94 % |
| 1 900 | 1 815.2 | 1 822.4 | 1 416.6 | 1 848.2 | **1.30×** | +1.42 % |
| 4 000 | 1 535.9 | **1 692.8** | 1 232.8 | 1 680.4 | **1.36×** | **−0.73 %** |

Output tokens/s, mean of the six measurement runs per cell (2 trials × 3 runs),
recomputed from `pr48944_replication/raw/{trial1,trial2}/phase4/<arm>/ctx_<N>/measure_*.json`.
At the 400 cell `static K=3` is 13.18 % faster than `C′` — a different number
from `C′` being 11.64 % below `static K=3`.

**Verdict, restated.** Two things the earlier text ran together:

1. **The pre-registered throughput criterion was met.** Against `A′`, three of
   four cells clear 1.15× and two clear 1.30×, under a rule frozen before
   measurement. That rule was a throughput-ratio rule only — it contained no
   dispersion term — so the incorrect dispersion figures discussed below did not
   enter the determination, though the verdict paragraph in the PR comment cited
   them as support. The criterion being met is unchanged and not withdrawn.
2. **A performance advantage from adding the context axis is not established by
   this campaign.** Against `static K=3` — a one-line schedule with no context
   axis — `C′` is lower at two cells and higher at two. The largest observed
   difference favours `static K=3`. Meeting a criterion set against `A′` does
   not carry a claim against `static K=3`.

How to read `A′`: in the 129–512 batch tier `A′` selects K=0, and that tier
covers the run's nominal concurrency — but `A′` still drafts, because
`num_prompts=256` with `max_concurrency=256` is a finite batch whose scheduled
size passes through the lower tiers on ramp-up and drain. Draft tokens summed
over the six runs: `A′` 1 243 / 1 720 / 5 748 / 10 183, `C′` 1 234 / 190 858 /
190 984 / 179 854, `static K=3` 197 043 / 202 695 / 208 644 / 190 812. This is
not a fixed-batch K=0-against-K=3 comparison and should not be described as one.

Dispersion, recomputed from the raw runs as the coefficient of variation over
the six measurement runs per cell (sample standard deviation, n−1): `C′`
1.19 / 1.90 / 1.70 / 1.74 %, `static K=3` 1.31 / 1.16 / 1.58 / 1.53 %, `A′`
1.18 / 1.10 / 1.02 / 0.53 %, `no_spec` 0.94 / 0.41 / 0.36 / 0.52 %.
**These are reported as observed spread and are not an equivalence test.** They
supersede the "stdev under 2 % in every cell" statement that stood here, which
fails for `C′` trial 1 at ctx 900 (2.02 %) under the per-trial definition, and
they supersede the dispersion figures in the PR comment, which reproduce under
neither definition. The "29–36 % signal against 0.7 % order bias" multiplier is
withdrawn: it was computed wrongly, and comparing the observed gain with an
order-associated difference does not establish statistical significance or
robustness.

**[Corrected, 2026-10-22.]** At prefix 400, the schedules prescribe the same K
in the high-batch tier and produced similar aggregate throughput. Per-step K
traces were not recorded. The earlier text here asserted that every step chose
the same K and read per-tier acceptance rates as confirmation of which tier
fired; neither is supported by what was recorded, and both conflict with the
ramp/drain behaviour described above.

**Against no-speculation.** `C′` reaches 0.70× / 0.94× / 1.02× / 1.09× of
`no_spec`. **[Correction, 2026-10-22]** This was previously headed "the
argument-free version" and read as independent support for the context axis. It
is not: `static K=3` reaches about 0.79× / 0.93× / 1.00× / 1.10× at the same
points, crossing in the same place and exceeding `C′` at 4 000. What the columns
support is the observation that **speculation's standing relative to
no-speculation changes with prefix length in this setup**. That is not
attributed to the context axis here, and it is not established as a general
property of K=3 either.

**What C does not show.** It is one model pair, one hardware configuration, one
concurrency (c = 256), and four context points. The eager control intended to
isolate the cudagraph contribution was attempted and discarded (ctx 4 000
exceeded the 4 096 budget; eager ctx 1 900 gave K3/K0 = 0.94×, the wrong
direction for a single-cause cudagraph hypothesis). **[Correction, 2026-10-22]** A "+0.6 %, therefore zero-cost when unused"
claim (2 627.2 against 2 642.3) stood here and is withdrawn. A single pair of
numbers inside run-to-run spread does not establish a zero-cost property, and
the cost of the unused 2-D form has not been measured.

**Earlier negative result, self-reported.** An earlier spec-bench aggregate
showed −4.0 % for a 2-D schedule. **[Correction, 2026-10-22]** The sentence
"the cause was using the ctx axis to *lower* K on the short-context majority"
is withdrawn as a causal attribution. That schedule did lower K on the
short-context majority, but the regression was never decomposed, so its cause is
not established. It is recorded as an observed regression for that
configuration.

---

## D — Sharpness of the K optimum at low batch

Sources: [`ctx_uplift/RESULTS.md` @ `08a9b59`](ctx_uplift/RESULTS.md) (2026-08-08)
and [`ctx_uplift/RESULTS_precision.md` @ `9f6e0b9`](ctx_uplift/RESULTS_precision.md)
(2026-08-13). b = 11, ctx 4 000 and 32 768, K ∈ {3, 5, 7}.

- **Direction holds, twice.** K\*(4k) = 3 against K\*(32k) = 5 on both runs.
- **Magnitude does not clear the bar.** Forcing loss max +4.51 % on the first run;
  on the precision re-run k5 against k3 is +3.60 % (+1.12σ) where the
  pre-registration required ≥2σ, and k3/k7 overlap at 1σ.
- **Verdict: WEAK, then CONCEDE.** The path "8k+ contexts, precision first" was
  closed on 2026-08-13. A third rental for more warmup was ruled out in advance
  and correctly so — warmup was already shown to work (the systematic cold-measure
  bias of the first run disappeared), and what remained was run-to-run variance.

### 2026-08-31 addendum — the residual variance has a named cause

vLLM #53436 (filed 2026-08-23, ten days after our concede) reports that at
temperature 0 with a fixed seed, speculative decoding produces identical output
text but large run-to-run throughput variance, concentrated at low concurrency,
because the target forward is not bit-reproducible: near-tie logits flip
accept/reject decisions, so accepted-tokens-per-step moves while the emitted
argmax does not. They measure acceptance against throughput at r = 0.98. vLLM
#54506 files the same mechanism as an RFC (verify computes rows with M = k+1,
plain decode with M = 1, and nothing guarantees those agree bit for bit).

Re-analysing our own archived raw (`ctx_uplift/precision_raw/ctx_precision_2026-08-12.tgz`,
9 measurement runs) against that claim:

| K | `num_drafts` across 3 identical runs | `acceptance_length` spread | throughput CV |
|---|---|---:|---:|
| 3 | 7 297 / 7 743 / 7 702 | 0.18 | 5.40 % |
| 5 | 6 431 / 6 348 / 6 311 | 0.06 | 2.40 % |
| 7 | 6 041 / 5 485 / 6 104 | **0.42** | 6.25 % |

Acceptance is not reproducible across identical runs on our stack either, and
within each K it tracks throughput: r = +0.794 (k3), +0.572 (k5), **+0.989** (k7);
pooled after removing the K effect, **r = +0.785**. Same sign and same mechanism
as #53436, at a lower coefficient on n = 3 per cell.

**What this changes, stated narrowly.** The attribution in `RESULTS_precision.md`
— that the residual is intrinsic variance of the b = 11 / 32k cell — is probably
wrong; there is a named engine-level cause. The *decision* it supported (do not
buy more warmup) remains correct, because warmup cannot fix forward-pass
non-determinism.

**What this does not change.** The effect size. +3.60 % is still +3.60 %. Only
the denominator of the significance test is implicated. A re-run becomes
justifiable only if the forward-invariance work lands and shrinks σ; #54506 is
open and proposes no design. **This is not a finding that the context axis is
larger than measured, and it must not be cited as one.**

---

## Corrections register

Kept here so the record is in one place rather than scattered across threads.

| Date | Correction |
|---|---|
| 2026-05-22 | FA #2592 disclosure rested on a `time.perf_counter` measurement error |
| 2026-06-14 | vLLM #45581 / #45583 closed — filed without the required AI-assistance disclosure |
| 2026-07-14 | `RESULTS.md` did not meet the linking project's language and voice norms |
| 2026-07-27 | `RESULTS.md` §2 ratios superseded — warm-up bias; direction holds, level attenuates |
| 2026-07-30 | RFC #48627 carried three factual errors |
| 2026-08-13 | Claim D downgraded WEAK → CONCEDE |
| 2026-08-24 | KV pool figures paired across a launch-order offset, understating the cost by ≈1 pp |
| 2026-08-24 | §4 labelled the pool loss as draft KV; it is an activation reservation. LongSpec citation withdrawn |
| 2026-08-24 | Qwen3-4B cell misread, self-reported |
| 2026-08-28 | §3 asserted the arms shared a prefix-cache hit rate; they do not |
| 2026-08-31 | C and D had been conflated in internal discussion; D's concede was being read as a verdict on C |
| 2026-09-01 | RFC #48627 §2 and its withdrawn ratios carried no supersession note in the body; added one |
| 2026-09-01 | #49986 still asserted the arms shared a prefix-cache hit rate; correction of record posted |
| 2026-09-01 | The MagicDec quotation in RFC #48627 and PR #48944 was attributed to arXiv:2408.11049, which does not contain it — it is from the MagicDec-part1 blog — and an em-dash clause had been dropped without marking the elision. Both bodies corrected |

---

## Prior art, verified at source

Checked against arXiv metadata and, where the claim is not in the abstract, the paper or blog text
itself. Listed because the argument that K needs a policy rather than a wider table rests on
several groups independently proposing *different* signals for the same decision.

| Work | ID | What it proposes | Verified |
|---|---|---|---|
| MagicDec | blog *MagicDec-part1*; paper [2408.11049](https://arxiv.org/abs/2408.11049) | speculation amount as a function of sequence length | Quote verbatim in the blog under *Adaptive Sequoia trees*. **Not in the paper** — the paper argues the same thing as a critical-sequence-length threshold plus an analytic model for the optimal drafting strategy |
| LibraSpec | [2608.08721](https://arxiv.org/abs/2608.08721) (2026-08-09) | speculative length by marginal gain | Title and framing confirmed: *Dynamic Diffusion-Based Speculative Decoding via Marginal-Gain-Driven Optimization* |
| EcoSpec | [2607.12696](https://arxiv.org/abs/2607.12696) (2026-07-14) | draft selection by marginal expert activation cost | Abstract confirms the mechanism and **"up to 1.62× speedup"** on DeepSeek-V3.1 / Qwen3-235B-A22B / GPT-OSS-120B. Per-model figures circulating elsewhere (1.50×, 1.47×) are **not** in the abstract |
| SparseSpec-L | [2607.27735](https://arxiv.org/abs/2607.27735) (2026-07-30) | speculation length by an online entropy-based controller | Abstract confirms *"An online entropy-based controller further selects the speculation length according to expected step-wise efficiency"* and that *"extending the speculation horizon can reduce rather than improve speedup when the marginal acceptance probability falls below the relative drafting cost."* The speedup figure is **elided in the abstract**; a "2.79×" figure circulating elsewhere could not be verified |
| Hybrid Verified Decoding | [2606.01019](https://arxiv.org/abs/2606.01019) (2026-05-31) | verification allocation learned from payoff | Title confirmed: *Learning to Allocate Verification in Speculative Decoding* |

Also in vLLM itself, not literature: #54082 proposes gating K between 0 and K on token entropy, and
the shipped schedule keys on batch size. So the signals actually being proposed for one decision are
batch, context, entropy, marginal gain, expert cost and learned payoff.

## What the ledger licenses

- **A is the strongest position**, because it is decomposed and has two
  independent external reproductions. Its weakness is that the batch scaling is
  not yet attributed to a term, and the run that would attribute it has not
  happened.
- **[Replaced, 2026-10-22] C met its pre-registered throughput criterion
  against `A′`.** This previously read that C "should not be described as
  unproven". This campaign does not establish a performance advantage over
  static K=3. Observed differences were −11.64 %, +1.94 %, +1.42 % and
  −0.73 %. The criterion being met against `A′` stands; it is not a result
  about `static K=3`. Breadth is a further limit: one model pair, one
  concurrency, four context points.
- **[Replaced, 2026-10-22] B and C do not jointly show that a context-blind
  schedule mis-picks K.** This previously concluded that they did. B shows
  speculation's economics move with context in this setup. C does not add that a
  schedule which cannot see context therefore chooses K badly, because an
  advantage over the context-blind `static K=3` was not established. What
  remains untested is whether a condition exists where a simple schedule does
  choose badly.
- **[Replaced, 2026-10-22] "The context axis pays where batch pressure is high"
  is withdrawn.** It was offered as the consistent reading of A, C and D
  together, and it was never tested as a hypothesis. The one campaign run under
  batch pressure (c = 256) did not establish it.

## What it does not license

- Any claim that the context axis delivers a fixed multiplier across workloads.
  **[Amended 2026-10-22]** This previously added "at ctx 400 it correctly does
  nothing". At prefix 400 the 2-D schedule prescribes K=0 in the high-batch tier
  and measured 11.64 % below `static K=3`, so that choice is not shown to be the
  correct one.
- Any use of the 1.17× / 1.25× / 1.38× / 1.36× figures.
- **[2026-10-22]** Any claim that the context axis outperforms a static or a
  well-tuned batch-only schedule. The 2026-07-27 campaign does not establish it.
  Nor does it establish the converse: `C′` was higher than `static K=3` at two
  of four points, by +1.94 % and +1.42 %.
- **[2026-10-22]** Any use of the dispersion figures published in PR #48944
  comment 5091663057, or of the "29–36 % signal" multiplier built on them.
- **[2026-10-22]** Any "zero-cost when unused" or "resource-free" statement
  about the context axis.
- Any claim that D's concede is reversed. It is not; only its stated cause is in
  question.
- Any citation of the survey literature compiled on 2026-09-01 for this
  reconciliation. Those paper-level numbers were not verified at source.
