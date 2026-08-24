# Results — ctx-scaling of the DSD baseline tax, mechanism attribution

**Run 2026-08-23, 1× H100 NVL (93.6 GB), vLLM 0.27.1, V1 runner
(`VLLM_USE_V2_MODEL_RUNNER=0`), transformers 5.10.2.**
Pre-registration: `PREREGISTRATION.md` (written 2026-08-13, amended 2026-08-23
before measurement). Raw: `run_20260823_gemma/` (primary),
`run_20260823/` (a secondary Qwen3 run, see §5).

Stack: target `prithivMLmods/gemma-4-31B-it-qat-FP8`, drafter
`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` — the same pair as
`tax_decomposition/`, so the two studies are directly comparable.

## 1. Headline

**The DSD baseline tax does not scale with context on this stack. It scales with
batch.**

| ctx | concurrency | no_spec TPOT | dsd_k0 TPOT | Tax | absolute |
|---:|---:|---:|---:|---:|---:|
| 400 | 2 | 14.374 ms | 15.317 ms | **+6.56 %** | +0.943 ms |
| 38 000 | 2 | 15.945 ms | 16.975 ms | **+6.46 %** | +1.030 ms |
| 4 000 | 26 | 27.218 ms | 28.945 ms | **+6.34 %** | +1.727 ms |
| 16 000 | 6 | 17.713 ms | 18.546 ms | **+4.70 %** | +0.833 ms |
| 400 | 189 | 77.164 ms | 93.820 ms | **+21.58 %** | +16.656 ms |

- **Primary endpoint — context, at fixed batch (C=2): +6.56 % → +6.46 % =
  −0.10 pp** across a 95× context range. Flat, and the sign is negative.
- **Batch, at fixed context (400): +6.56 % (C=2) → +21.58 % (C=189) = +15.02 pp**
  across a 90× batch range.

The context and batch axes had to be separated by hand. The KV pool couples
them — with the drafter loaded, concurrency 189 fits at ctx 400 but only 2 at
ctx 38 000 — so a sweep that lets concurrency follow context mixes a +15 pp
batch effect into the context reading. The fixed-batch column (C=2) is what
isolates context; the production-shaped cells are reported but are not the
endpoint.

The per-cell concurrency was set from the pool measured in a dedicated probe
launch before the sweep (`kv3.log`, 125 435 tokens), using the harness rule
`concurrency = floor(0.9 * pool / (ctx + 197))`, where 197 is the request
suffix this workload adds (`--prefix-repetition-suffix-len 96`,
`--prefix-repetition-output-len 100`, plus one). That reproduces the four
concurrencies used — 189, 26, 6, 2 — exactly. Note this probe figure is not
the same launch as the pool numbers in §4, which are what the sweep's own
servers reported; see the launch-order note there.

## 2. Replication of `tax_decomposition/`

At ctx 4 000, V1, K=0: **+6.34 % here vs +7.29 % on 2026-08-05** — different
vLLM version, different session, different concurrency, ~1 pp apart. The two
measurements corroborate each other.

## 3. Mechanism

The tax is charged with **zero draft tokens produced**. Summed over every
measured run of the spec arm:

```
spec_decode_num_draft_tokens_delta = 0.0
```

Runtime K=0 is honoured on V1 (`gpu_model_runner.py` threads
`scheduler_output.num_spec_tokens_to_schedule` into `propose()`), so this is not
the MRV2 bug of #51510. Workload is identical between arms (generation 4800,
prompt 23808, prefix-cache hits 23040 per run at the anchor cell), preemptions
are 0 in both, and the per-step token census is nearly identical (816 vs 818
steps, both dominated by the decode-only bucket). The extra time is per step,
not extra steps.

What differs is the CUDA graph mode. vLLM says so itself:

```
WARNING [vllm.py:865] Dynamic speculative decoding changes the target verification
length at runtime. Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE
for reliability. Use VLLM_USE_V2_MODEL_RUNNER=1 if you want to use full CUDA graphs.
```

| arm | captured |
|---|---|
| `no_spec` | PIECEWISE=51 **and FULL=51** |
| `dsd_k0` | PIECEWISE=51 only |

Enabling dynamic SD costs the decode path its full CUDA graphs even when the
schedule never asks for a draft token.

**The escape hatch the warning names is the one with the bug.** V1 honours K=0
but is forced to PIECEWISE; MRV2 keeps full graphs but ignores the scheduler's
K and runs the full draft pipeline anyway (#51510, fix in PR #51575). There is
currently no configuration that gives dynamic SD without one of the two.

## 4. Drafter KV cost (H-kv)

Same box, same flags throughout — every launch in this study ran at
`gpu_memory_utilization=0.9` and `max_model_len=52224`, and the only
configuration difference between arms is the presence of `speculative_config`.

The profiled pool is not identical across launches, and the pattern is
systematic rather than random. Ordered by launch time:

| launch | arm | GPU KV cache size |
|---|---|---:|
| 11:52 `kv3` probe | `dsd_k0` | 125 435 tokens |
| 11:57 `ctx400_c2` | `no_spec` | 140 459 tokens |
| 12:03 – 12:42, five launches | `dsd_k0` | 127 021 tokens (all five) |
| 12:08 – 12:38, four launches | `no_spec` | 142 046 tokens (all four) |

**The first launch of each arm is low by 1 586–1 587 tokens, and every launch
after it is stable to the token.** The same offset appearing in both arms rules
out random profiling noise; the cause is not established here, container start
state being the obvious suspect.

That matters for the pairing. Comparing like launch to like launch:

| pairing | drafter cost |
|---|---:|
| both stable (142 046 − 127 021) | **15 025 tokens (10.6 %)** |
| both first-launch (140 459 − 125 435) | **15 024 tokens (10.7 %)** |

Two independent pairings agree to one token. **≈15 025 tokens — about 10.6 % of
the pool — are spent on a drafter that never drafts.** By the harness's own
sizing rule that is concurrency 189–191 with the drafter against 211–214
without it, so roughly 22 concurrent requests at this workload's shape.

(An earlier version of this section paired `no_spec ctx400_c2` with the `dsd_k0`
sweep pool and reported 13 438 tokens / 9.6 %. Those are both real log values
from the same cell, but `no_spec ctx400_c2` is that arm's first launch while the
`dsd_k0` cell is its second, so the pairing crossed the launch-order offset and
understated the cost by about 1 pp.)

This is the concrete form of the drafter-KV problem LongSpec
([2502.17421](https://arxiv.org/abs/2502.17421)) answers with a constant-size
draft cache. It does not show up in TPOT; it shows up as concurrency you cannot
have, which is why the batch axis above matters.

## 5. Why a second stack is in the raw, and what it is worth

`run_20260823/` holds an earlier sweep on Qwen3-4B + Qwen3-0.6B. It exists
because the harness asked for `"method": "mtp"` with no drafter, which cannot
work — the Gemma target carries no MTP head, and Gemma-4 MTP is an MRV2 feature
regardless. The harness was wrong; the runbook's model pair was right. The
correct fix (drafter model, no `method` key, exactly as `tax_decomposition`
does it) is what §1 runs.

That Qwen3-4B sweep reported +82.7 % → +91.2 % and **appeared to show the tax
scaling with context (+8.49 pp)**. It does not survive contact with the original
stack. The reconciliation is that the PIECEWISE penalty is charged against the
step, so its share depends on what a step costs:

| stack | step cost, short → long ctx | tax |
|---|---|---|
| Qwen3-4B, C=6 | 4.29 → 11.84 ms (2.76×) | grows with ctx |
| Gemma-4-31B, C=2 | 14.37 → 15.95 ms (1.11×) | flat |

A 4 B model at long context is attention-bound, so the step — and with it the
penalty — grows. A 31 B model at low batch is weight-bound, so neither does.
The percentage is a property of the denominator, not of context. Reported here
so the number is not mistaken for a result about context.

## 6. Cross-stack read on #49986

Suppressor72 measured the tax widening with context on dual RTX 5090 / MRV2:
−8.5 % (short) → −35 % (32 k), a 4.1× change in the fraction. **This stack does
not reproduce that shape** — 6.56 % → 6.46 %, 1.0×. Per the pre-registered
honesty gate, that is reported as a negative cross-stack result, not softened.

The pre-registration committed in advance to what a flat result would mean: the
leading explanation for their scaling is then the MRV2 wasted-draft bug, whose
cost does grow with context because a discarded draft step still attends over
the whole KV. Their runner is MRV2 (`VLLM_USE_V2_MODEL_RUNNER=1`, stated in
#51510). We cannot confirm that from here; it is a lead for their rig, and
PR #51575 is the thing to test it with.

What this stack does say is that a **second, structural tax exists independently
of that bug** — visible on V1 with K=0 honoured — and that it is driven by batch
rather than context.

## 7. Limits

- One GPU, one target model, one drafter, one workload shape
  (`prefix_repetition`, 100 output tokens). Median of 3 measured runs after 1
  warm-up per cell; no confidence intervals are claimed.
- Context and batch cannot be varied independently beyond what the KV pool
  allows; the fixed-batch column is C=2, which is a low-batch regime.
- `+21.58 %` at C=189 is a single cell at one context. The batch axis deserves
  its own sweep before any claim about a threshold — cf. #49548, which reports a
  far larger collapse on that axis.
- The tax is measured at K=0. It is the cost of having dynamic SD enabled, not
  the cost of speculating.
- The profiled KV pool carries a first-launch offset of ~1 586 tokens (§4). Any
  future run that compares pools — in particular a `num_gpu_blocks_override`
  arm sized against a measured pool — has to discard the first launch of each
  arm, or match launch positions, or it will be reading this artifact.
