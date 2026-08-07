# Pre-registration — Context-length uplift sweep: the batch-only representation cannot express the optimal K

**Status:** PRE-REGISTRATION. Locked before rental. §1 Thesis, §5 Success criteria, §7 Judgment matrix are frozen — no post-hoc edits after data is read (discipline from the tax-decomposition and P1 campaigns).
**Date:** 2026-08-07
**Cost ceiling:** $10 (hard). H100 NVL 94 GB. Estimated ~3.4 h ≈ $6.8 (12 spine arms, no order-reversal — see §6.4).
**Vehicle:** Gemma-4-31B QAT FP8 (weight quant; **KV cache is bf16**, see §3), hybrid attention + MTP + DSD. Config of record: `benchmarks/configs/gemma_4_31b_qat_fp8_dsd.yaml`. **Engine + build SHA = match `pr48944_replication` / `tax_decomposition` stack (pin at rental)** *except* `--max-model-len` is raised 8192 → **33024** (required to reach 32k prompts). This deviation is recorded; the 4k anchor (§9) verifies the stacks are otherwise comparable.

---

## 1. Thesis (frozen)

Not "the shipped `dsd_schedule` table is mistuned" — that is defeated by *"just retune it."* The claim is about the **representation**, not any particular table:

> **No batch-only assignment `K(batch)` can be simultaneously optimal at short and long context.** At a *fixed* batch, the throughput-optimal K is *contradictorily* required by context length: `K*(b, 4k) ≠ K*(b, 32k)`. Therefore a batch-only table is not mistuned — it is **inexpressive**. Context length must be an explicit axis of the schedule.

Strongest form (if reached): the optimal **K ceiling itself rises with ctx** — at long ctx the memory-bound decode makes additional draft tokens nearly free, so `K*` climbs above the schedule's max assigned value (3).

Corollary this refutes: "the cost model already contains ctx implicitly (TurboSpec Eq 3 `N_context`, SGLang offline cost table)." An implicit cost term scores how expensive a K is; it does not give the operator a surface to **declare** K per regime, and a single `K(batch)` value cannot be right in two regimes at once regardless of how the cost model scores it.

## 2. Why this framing (design rationale)

- **Kills the retune rebuttal.** A contradiction at fixed batch is table-independent: *whatever* single K the table assigns to batch `b`, it is ≥15% suboptimal at one of {4k, 32k}.
- **Kills tier-collapse.** We never need to reach high batch at 32k. The spine lives at a batch reachable across *all* ctx (§3), which is low → cheap per arm → friendly to $10.
- **Metric is intrinsic.** We measure *forcing loss* (impose one ctx's optimal K on the other), a property of the representation, not of a shipped artifact.

## 3. KV pool → reachable batch (bf16 KV — corrected)

**KV cache dtype is bf16, not fp8.** The model's FP8 is *weight* quantization; the engine runs `dtype=torch.bfloat16, kv_cache_dtype=auto` (→ bf16). `--kv-cache-dtype fp8` is **prohibited on this stack** (FlashInfer SM90 + sliding-window drafter guard, #48495; recorded in `tax_decomposition/RUNBOOK.md:67`). So KV is **2 B/elem**, and reachable batch is ~½ of an fp8 assumption.

Arch (`config.json` text_config): 60 layers = 10 full_attention (`num_global_key_value_heads=4`, `global_head_dim=512`) + 50 sliding_attention (`num_key_value_heads=16`, `head_dim=256`, window=1024). `attention_k_eq_v=True` (may or may not halve the stored cache — the §6.1 gate measures which).

Per-request KV (bf16, sliding layers capped at 1024):

| ctx | KV/req (bf16, no halving) | est. max batch | (bf16, k_eq_v ½) est. max batch |
|---:|---:|---:|---:|
| 4k  | 1.15 GB | ~37 | ~74 |
| 8k  | 1.48 GB | ~29 | ~58 |
| 16k | 2.14 GB | ~20 | ~40 |
| **32k** | **3.44 GB** | **~12** | **~24** |

**Measured calibration (verified):** at `--max-model-len 8192 --gpu-memory-utilization 0.90` this exact stack reported `kv_cache_size_tokens = 49,375`, `kv_cache_max_concurrency = 6.03` (full-length-8192 seqs) — nets out target + MTP-draft + overhead. (`tax_decomposition/raw/.../phase0_v1_dsd_k0_metrics.txt`.)

**Caveat (do not over-claim):** vLLM's hybrid KV pool accounting may treat the sliding-window layers differently from the per-request model above. The numbers here are **estimates**; the §6.1 gate measures `kv_cache_max_concurrency` and actual OOM directly at mml=33024 to fix `b`. **Provisional spine `b ≈ 11`** (bf16, no halving, 90% of ~12); if the gate confirms k_eq_v halving, `b ≈ 22`.

## 4. The spine (minimal decisive experiment)

Fixed batch `b` (§3, gate-confirmed), two ctx endpoints × extended K:

- **ctx ∈ {4096, 32768}** — 4k = anchor band; 32k = long-ctx endpoint (mandatory, an endpoint of the spine, not secondary).
- **K ∈ {0, 1, 2, 3, 5, 7}** where **K=0 ≡ DSD-tier K=0** (the schedule *can express* this; it carries the 7–17% baseline tax measured in #49986). This is the canonical K=0 for the thesis.
- **+ one `no_spec` reference arm per ctx** (`speculative_config` absent) — the true baseline. Distinct from DSD-tier K=0; if `K*=0` we must know which zero it is. `no_spec` also serves the §9 anchor.
- **12 spine arms** (2 ctx × 6 K) at fixed `b`, + 2 `no_spec` reference arms. This proves or disproves the thesis.

**Extension (only if budget remains under $10):** ctx ∈ {8192, 16384} to fill the curve; a second fixed batch `b'` for a confirmatory contradiction. Extension does not gate the verdict.

Workload: `completion_tokens ≥ 256` at **all** ctx cells (do **not** shorten output at high ctx — a 32768-prefix + short output collapses the decode steady-state window and inflates TPOT noise). duration 120 s, warmup 30 s.

## 5. Metric & success criteria (frozen)

Per cell: **median output token throughput** (tok/s at fixed concurrency `b`); secondary key = median TPOT. `n = 3` per cell.

- `K*(c) = argmax_K throughput(K, c)` for `c ∈ {4k, 32k}` (K over the DSD-tier sweep {0,1,2,3,5,7}).
- **Forcing loss:** `L(c_src → c_dst) = 1 − throughput(K*(c_src), c_dst) / throughput(K*(c_dst), c_dst)`.

**PASS (thesis supported):** `K*(4k) ≠ K*(32k)` **AND** `max( L(4k→32k), L(32k→4k) ) ≥ 15%`.
The 4k→32k direction (short-ctx-optimal K imposed at 32k) is the primary indictment of a short-ctx-tuned batch-only table.

## 6. Rental-start gate (before any measured arm)

1. **Fix `b`.** At mml=33024, read `kv_cache_max_concurrency`; confirm whether `attention_k_eq_v=True` halves the cache; run one 32k request at candidate `b` as an OOM smoke test. Set `b` = 90% of measured 32k max concurrency.
2. **Confirm dtype.** Verify `kv_cache_dtype=auto`→bf16 in the engine log; do **not** pass `--kv-cache-dtype fp8` (#48495).
3. **Confirm K=0 semantics.** Check `num_speculative_tokens: 0` passes config validation (DSD-tier K=0 arm); if not, realize it via the schedule and note the mechanism.
4. **Controls.** Pin build SHA; cold-start burn; `PYTHONHASHSEED=0`; fixed request seed; single build for all arms; one spot re-check at the end for thermal drift.
   **Order-reversal is omitted** (cost: it would push 12 arms from ~3.4 h to ~5.5 h ≈ $11, over ceiling — K is launch-time config, one server restart per arm). It is not needed at this effect size, and the justification uses two *separately-sourced* bounds for two *distinct* confounds:
   - **Position/order bias ≤ 1.72%** — a full order-reversed replication on the same rig and methodology (PR #48944 Phase 4, trial 1 vs trial 2 at reversed arm positions) bounds it; max across 4 arms × 4 ctx, recomputed from published raw at `pr48944_replication/raw/{trial1,trial2}/phase4/` (e.g. `static_k3/ctx_1900`: T1 1806.8 → T2 1837.9 tok/s). This is the measurement that actually licenses skipping order-reversal.
   - **Session drift ≈ 0.39%** — separately bounded by the tax-decomposition `no_spec` pair measured ~2 h apart (`tax_decomposition/RESULTS.md:88`); this is a drift-plus-runner-baseline bound, not pure order bias.

   Both are an order of magnitude below the 15% success bar. A skeptical reader can recompute the 1.72% directly from our published raw. A single end-of-session spot re-check is retained (item 4 above).

## 7. Judgment matrix (frozen)

| Outcome | Meaning | Report |
|---|---|---|
| `K*(4k)≠K*(32k)` & forcing-loss ≥15% | ctx axis **necessary for expressiveness** | PASS → RFC #48627 evidence ladder (headline) |
| `K*(4k)≠K*(32k)` & forcing-loss <15% | K differs but penalty small at this `b` | WEAK → report honestly; try `b'` or concede point |
| `K*(4k)=K*(32k)` | single `K(b)` suffices at this batch | NULL → thesis unsupported here; publish anyway (no post-hoc reframe) |
| `K*(32k)=7` (sweep ceiling) | optimal ≥7; forcing-loss is a **lower bound** | CEILING → strongest thesis form (ceiling rises with ctx); extend sweep, report as lower bound |

## 8. Anticipated rebuttals (pre-registered refutations)

1. **"The table is a config — just retune it."** → Invalid by construction: the spine shows a *contradiction at fixed batch* (`K*(4k)≠K*(32k)`), so no single retuned `K(b)` is simultaneously optimal. Expressiveness, not tuning.
2. **"That low batch isn't a production regime."** → At 32k the *physically reachable* max concurrency on 94 GB is ~12 (§3, bf16). The regime is not a choice; it is what the memory budget permits at long ctx — precisely the per-user-latency-dominated, speculation-is-the-lever regime MagicDec describes.

## 9. Anchor & scope

- **Anchor (replaces an unachievable abort condition).** The 1.29–1.36× band is `C′/A′` (2D vs batch-only) @ concurrency 256 — a *different* measurement than this fixed-K sweep @ batch ~11, so it cannot be reproduced here. Instead: the **4k `no_spec` vs DSD-tier-K=0** delta must fall in the **#49986 tax band (~7% on V1)**. That is the verifiable comparability check for "same stack." If it does not, the stack differs (record and re-scope).
- `max_model_len` deviation (8192 → 33024) is recorded; it is required for 32k and does not affect the 4k anchor's comparability (verified by the tax-band check).
- We do **not** claim the shipped `dsd_schedule` values are wrong; we claim no batch-only *shape* covers {4k, 32k} at fixed batch.
- Single vehicle, single build. K swept as fixed-K arms (schedule bypassed) to isolate `K*` per cell.
