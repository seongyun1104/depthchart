# ctx-uplift RUNBOOK — execution-only

Pre-registration: `ctx_uplift/PREREGISTRATION.md` (commit `825abd5`, frozen before data).
Vehicle: Gemma-4-31B QAT FP8 (weights) + MTP + DSD, **KV=bf16**. Rental is execution
only; all design decisions are in the pre-reg. Cost ceiling **$10**. Est. ~1.9 h
(6 servers × 2 ctx) + build/download.

## 0. Provision
- Vast H100 **NVL 94 GB** (not 96 — [[h100-nvl-vram-94gb]]). Pin instance.
- **HF auth first** (`hf auth token`, account jamesyun) — unauth pull is ~8× slower
  ([[feedback-gpu-time-is-money]]). Pull target + draft to `/root/models/{target,draft}`.

## 1. Build (match the anchor stack)
- vLLM: **same build SHA as `pr48944_replication` / `tax_decomposition`** (pin; record).
  `VLLM_USE_PRECOMPILED=1 pip install -e .` per the verified recipe.
- Record: `git -C $VLLM_REPO rev-parse HEAD` → the harness writes `build_sha_{arm}.txt`.
- **Deviation from anchor stack:** `--max-model-len 33024` (was 8192; required for 32k).
  Recorded in pre-reg §9; verified benign by the 4k anchor tax-band check (§6 below).
- **Do NOT** pass `--kv-cache-dtype fp8` (#48495 FlashInfer SM90 + sliding-window
  drafter guard). KV stays bf16. `export FLASHINFER_DISABLE_VERSION_CHECK=1`.

## 2. Gate — fix `b` from the engine's own KV log (no probe rental)
```
RESULTS_DIR=/root/results VLLM_REPO=/workspace/vllm python master_bench_ctx.py gate
```
- Launches ONE spec-on server (k1, so draft VRAM is reflected), parses
  `GPU KV cache size` + `Maximum concurrency for 33024 tokens per request: Y.YYx`.
  With mml=33024 and 32k req = 32768+256, that concurrency IS the 32k reachable batch —
  authoritative over the ~11/12 hand-calc (engine accounts the hybrid sliding cap).
- `export SPINE_B=<suggested b>` (= floor(0.9 × concurrency)).
- **If b is far from ~11/12, record the fact** (`kv_capacity_k1.txt`) — it validates or
  corrects our KV accounting (pre-reg §3).

## 3. Verify K semantics BEFORE the sweep (K is the whole experiment)
- After each server is up, the harness writes `spec_config_{arm}.txt` from the engine's
  `SpeculativeConfig(...)` line. **Check:**
  - `k0`: schedule `[[1,512,0]]` present → DSD-tier K=0 (the #49986 tax path).
  - `k1..k7`: `num_spec_tokens=K`, **no** `_per_batch_size` key.
  - `k5`,`k7`: confirm the MTP drafter actually drafts that many (`spec_config` line +
    `spec_decode_num_draft_tokens / num_drafts` in the snapshots). If MTP caps below 7,
    that cap is itself a finding (record; the CEILING verdict still applies).
- 32k OOM smoke: the harness runs burn at `b` first; if it OOMs, lower `SPINE_B` and note.

## 4. Sweep (6 servers, both ctx each = 12 cells)
```
SPINE_B=<b> RESULTS_DIR=/root/results python master_bench_ctx.py k0 k1 k2 k3 k5 k7 no_spec
```
- Each server benches ctx {4096, 32768}, 3 warmup discarded + 3 measure per ctx.
- `no_spec` is the 4k anchor reference (also benched at 32k for completeness).
- Per-ctx `num_prompts` is computed from `b` and recorded in each result JSON.

## 5. Spot re-check (end-of-session drift bound)
```
SPINE_B=<b> RESULTS_DIR=/root/results python master_bench_ctx.py spot
```
- Re-measures the k0 reference at both ctx into `grid/spot/`. Compare to the first k0
  run; large drift ⇒ flag (order-reversal was omitted, justified by ≤1.72% + 0.39%,
  pre-reg §6.4 — this spot bounds the residual).

## 6. Aggregate — run it NOW, not just at desk (tax lesson: no ornamental gates)
```
RESULTS_DIR=/root/results python aggregate_ctx.py
```
- Prints per-cell throughput, `K*(4k)`/`K*(32k)`, forcing loss both directions, the §7
  verdict, and the **4k anchor tax-band check** (no_spec vs DSD-K0 must land ~7–17%).
- **If the anchor is out of band, the stack is not comparable — record and re-scope
  before trusting the verdict** (this is the abort gate that actually fires here, since
  the 1.29–1.36× band cannot be reproduced by a fixed-K sweep).

## 7. Teardown
- `tar czf ctx_uplift_$(date +%F).tar.gz -C /root results` → scp to desk.
- **Destroy the instance** and confirm (wall-time = real money, no salary backstop).

## Discipline
- depthchart commits: identity `seongyun.kim` (local config), no Co-Author, push then
  rider B ([[feedback-git-identity-verify]]).
- Rotate HF token after the rental ([[reference-hf-token]]).
- Results (RESULTS.md + raw grid + server logs + kv_capacity/spec_config/build_sha) land
  in `ctx_uplift/`, self-contained, before any public claim.
