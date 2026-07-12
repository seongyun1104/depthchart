# speculative-decoding-lab

Single-GPU experimentation lab for the **(B, ctx) context-aware K controller**:

> Speculative decoding's optimal draft depth K is a function of both batch size B and decode-time context length ctx. The batch axis is already dynamic in production (vLLM 0.24 DSD, SGLang adaptive). The context-length axis is unclaimed in both frameworks and the literature. Long-context decode amortizes KV-read across K speculatively drafted tokens, keeping K > 0 optimal well past the batch-only crossover point where fixed K collapses. This lab measures the (B, ctx) surface, replicates it across two engines, and produces the controller that turns the surface into a runtime policy.

**Two parallel tracks, one harness.** The infra (`scheduler/vllm_runner.py`,
`speculative/adapter.py`, `benchmarks/runner.py`) is shared. The tracks
differ only in the model / drafter / GPU-budget triple:

| Track                | Main model                              | Drafter                                | GPU target       | Status                    |
|----------------------|-----------------------------------------|----------------------------------------|------------------|---------------------------|
| Same-graph MTP head  | `LGAI-EXAONE/EXAONE-4.5-33B-FP8`        | (native in-graph MTP layer)            | H100 96 GB       | deferred (drafter MAL ~1.5, low detection power) |
| External MTP drafter | `AxionML/Gemma-4-12B-NVFP4`             | `google/gemma-4-12B-it-assistant`      | RTX 5090 32 GB   | secondary                 |
| External MTP drafter | `prithivMLmods/gemma-4-31B-it-qat-FP8`  | `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` | H100 96 GB | primary (campaign) |

Multi-instance, PD-disaggregation, and llm-d remain out of scope.
LMCache is demoted to a capacity track (orthogonal to the DSD core);
see `lmcache/` for that arm.

## Thesis

Two independent axes drive the optimal K:

```
B ↑ (short ctx, fixed drafter) → verify cost dominates
                                → optimal K drops toward 0 at the batch-only crossover
ctx ↑ (fixed B, fixed drafter) → per-token KV-read grows with ctx (memory-bound decode)
                                → spec K amortizes that KV-read across K drafted tokens
                                → K > 0 stays optimal well past the batch-only crossover
```

The (B, ctx) surface — not B alone — is the schedule table this lab
produces. That table drops into vLLM 0.24 DSD's
`num_speculative_tokens_per_batch_size`, extended by a ctx-length axis
that is not in the framework today, and is replicated on SGLang for
engine-independent evidence.

## Layout

```
speculative/           MTP / EAGLE / external-drafter adapters; acceptance hooks
lmcache/               LMCache configs (cpu_only, cpu_disk, cpu_redis); hit-rate workload gen
scheduler/             vllm_runner.py (primary); sglang_runner.py kept as reference
benchmarks/            sweep harness (B × ctx × K), metric collector; hit_rate is a covariate
benchmarks/configs/    per-model YAML: exaone_4_5_33b_fp8, gemma_4_12b_nvfp4, gemma_4_31b_qat_fp8
kv-transfer/           (out of P1 scope; placeholder for future)
papers/                REFERENCES.md — citations and gap mapping
```

## Stack

- Engine: vLLM OpenAI-compatible server, launched via
  `scheduler/vllm_runner.py` with:
  - `--speculative-config <dict>` (see two routing paths below)
  - `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
  - `--enable-chunked-prefill`, `--max-num-batched-tokens`, TP=1
- Two spec routing paths, chosen automatically by
  `benchmarks/runner.py::run_one` based on `cfg.model.draft_model`:
  - **Same-graph MTP head** (EXAONE 4.5):
    `'{"method":"mtp","num_speculative_tokens":K}'`
    — vLLM routes through the EAGLE proposer path against the model's
    own MTP layer.
  - **External MTP drafter** (Gemma 4):
    `'{"method":"mtp","model":"google/gemma-4-<size>-it-assistant","num_speculative_tokens":K}'`
    — `method:"mtp"` is required. Without it, vLLM silently falls back
    to the generic draft-model path on versions that pre-date the
    `gemma4_mtp` registration in `MTPModelTypes`; that fallback breaks
    target/drafter state sharing and changes the physical quantity we
    measure. `scheduler/vllm_runner.py::_assert_spec_routing()` parses
    the startup `SpeculativeConfig(method=...)` line and raises if the
    requested method is not what the engine actually loaded, so a
    version regression fails fast instead of silently corrupting the
    sweep.
- Adapter: `speculative.adapter.for_exaone_45()` /
  `for_gemma_4(k, draft_model)`; both return a `SpecConfig` whose
  `to_vllm_speculative_config()` produces the correct dict shape
  (Gemma 4 emits `method` **and** `model` together).
- KV layer: LMCache via `LMCACHE_CONFIG_FILE` env var
  (Phase 1 = HBM + CPU; Phase 2 = HBM + CPU + local NVMe)
- Bench: async harness in `benchmarks/runner.py`
  (streaming `/v1/chat/completions` + Prometheus `/metrics` polling)

## Fixed workload knobs

The harness pins these on every request to keep throughput comparable
across (B, ctx, K) cells and across models:

- `chat_template_kwargs.enable_thinking: false` — EXAONE 4.5's hybrid
  reasoning template inflates completion length ~21% when thinking is
  on, which contaminates the throughput signal. Harmless for Gemma 4.
- `ignore_eos: true` + `max_tokens: 256` — normalise decode length so
  cells are comparable at fixed output cost.
- `temperature: 0.0` — greedy decode; verify path determinism.

### Measurement window (canonical protocol)

Two 40s / 45s window-oversight incidents in prior campaigns motivated
these defaults:

- `warmup_s: 30` — first 30 s discarded so all cells enter steady state
  (wave-ramp equilibration, prefix-cache warmup, LMCache retrieve
  path if arm=lmcache) before any tokens are counted.
- `duration_s: 120` — the post-warmup measurement window. Long enough
  to average over per-step scheduler jitter observed in the DSD
  boundary probes (§ MTP × DSD probe: c=192–256 per-step scheduled
  requests oscillate around the tier boundary).
- Throughput: **Prometheus counter delta is canonical.**
  `RunResult.throughput_tok_s()` returns
  `Δ(vllm:generation_tokens_total) / Δwindow` over the snapshots that
  fall inside `[started_ts, ended_ts]`. Client-side per-request
  completion sums are still recorded as `throughput_tok_s_client()`
  for cross-check; when the two disagree, the counter wins and the
  gap flags an in-flight tail at window close.
- Acceptance rate: `Δ(vllm:spec_decode_num_accepted_tokens_total) /
  Δ(vllm:spec_decode_num_draft_tokens_total)` on the same snapshots.
- Preempts / KV usage / running queue depth: sampled by the
  Prometheus poller (`interval_s=5.0`) as pressure covariates for the
  overload branch of the K controller.

## Sweep design

Each track uses its own YAML. Cell counts are trimmed to fit VRAM.

| variable     | EXAONE 4.5 33B FP8 / H100         | Gemma 4 12B NVFP4 / RTX 5090      | Gemma 4 31B QAT-FP8 / H100        |
|--------------|-----------------------------------|-----------------------------------|-----------------------------------|
| batch size B | 1, 4, 16, 32, 64, 128, 192, 256   | 1, 4, 16, 32                      | 1, 4, 16, 32, 64, 128, 192, 256   |
| ctx tokens   | 256, 512, 1024, 2048, 4096        | 256, 512, 1024, 2048              | 256, 512, 1024, 2048, 4096        |
| spec K       | 0 (ref), 1, 2, 3                  | 0 (ref), 1, 2, 3                  | 0 (ref), 1, 2, 3                  |
| spec method  | `mtp` (same-graph)                | `mtp` (external drafter)          | `mtp` (external drafter)          |
| drafter      | (native)                          | `google/gemma-4-12B-it-assistant` | `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` |
| max_context  | 32768                             | 16384                             | 32768                             |

`hit_rate` and `hit_source` are workload covariates (recorded per cell,
not swept). Default `hit_rate=0.0, hit_source=apc`. LMCache runs live
in a separate config (see `lmcache/`) — the DSD-core sweep is APC-only.

Metrics: TTFT, ITL, throughput, MTP acceptance rate per (K, B, ctx).
Covariates recorded per cell: `ctx_tokens`, `hit_rate_target`,
`hit_source`, `kv_pool_tokens` (parsed from the vLLM
`GPU KV cache size: N tokens` line), `drafter_loaded` (True for K ≥ 1),
`spec_method_logged`, `enable_prefix_caching`, `enable_lmcache`,
`prefill_share` (Prometheus prompt/generation token delta over the
measurement window).

### K=0 semantics and the DSD baseline

Note that K=0 in this sweep means *no speculative config emitted* —
drafter weights are not resident in VRAM. That is not the same as vLLM
0.24 DSD's `num_speculative_tokens_per_batch_size[...]=0`, which keeps
the drafter loaded and only skips the draft/verify at scheduling time.
Consequences for analysis:

- The K=0 cell is a **reference point** (no-spec, no-drafter). It is
  not the correct baseline for measuring K's marginal cost, because
  the KV pool size is larger without the drafter weights + drafter KV.
- The **DSD-portable baseline** for "adding K" is K=1, which has the
  drafter loaded. Compare K=2, K=3 against K=1, not against K=0.
- To make the pool-size confound auditable ex post, every cell records
  `kv_pool_tokens` and `drafter_loaded`; the per-cell delta is a
  covariate the analysis pipeline can adjust for.

## Memory budget (`gpu_memory_utilization = 0.85`)

### H100 96 GB (Hopper `sm_90`, FP8 tensor cores)

**EXAONE 4.5 33B FP8** (same-graph MTP head):

```
96 GB × 0.85 = 81.6 GB  (weights + KV pool budget)
weights ≈ 33 GB (FP8) → KV pool ≈ 48.6 GB
MTP head shares the same graph; no separate drafter VRAM
reserve = 96 - 81.6 = 14.4 GB
```

**Gemma 4 31B QAT-FP8** + `gemma-4-31B-it-qat-q4_0-unquantized-assistant` drafter:

```
96 GB × 0.85 = 81.6 GB  (weights + KV pool budget)
weights ≈ 31 GB (FP8) → KV pool ≈ 50.6 GB
drafter resident: ~4 GB (BF16 assumption)
reserve = 96 - 81.6 = 14.4 GB
```

The 14.4 GB reserve is conservative (rule of thumb is 5-8 GB) to absorb
MTP draft-tree activation spikes and LMCache transfer buffers. After
the first sanity run, inspect free VRAM reported by vLLM at startup and
tune `gpu_memory_utilization` up toward 0.90 if there's slack.

### RTX 5090 32 GB (Blackwell `sm_120`, NVFP4 native)

**Gemma 4 12B NVFP4** + `gemma-4-12B-it-assistant` drafter — realistic
budget updated from Vast.ai 5090 measurements (see project memo
`p1_vast_install_recipe.md`), not the naive weights-plus-KV formula:

```
32 GB × 0.85 = 27.2 GB  (weights + activations + KV pool budget)
weights ≈ 12B × 0.5 B/param (FP4)  =  6 GB
CUDA graph capture (FULL_AND_PIECEWISE, MTP head)
                     + NVFP4 scales + activations ≈  6 GB
drafter resident (BF16)                            ≈ 1-2 GB
  → realistic KV pool ≈ 27.2 - 6 - 6 - 2 = 13.2 GB
reserve = 32 - 27.2 = 4.8 GB
```

The earlier ~21.2 GB estimate ignored FULL_AND_PIECEWISE capture
memory and NVFP4 activation scales; the ~13 GB figure is what actually
gets reported in the `GPU KV cache size` startup line and is what the
covariate `kv_pool_tokens` will pin per cell. Trigger the workaround
below when the observed pool drops toward 13 GB, not toward 21 GB.

Notes:
- Blackwell `sm_120` supports FP4 tensor cores natively; NVFP4 keeps
  matmul on-tensor-core without dequant overhead.
- If the drafter can't co-reside with the KV pool at hit ≥ 90%, drop
  `chunked_prefill_size` from 4096 → 2048 or lower
  `gpu_memory_utilization` to 0.80.
- **LMCache × `kv_cache_dtype=fp8`**: LMCache's KV store/load path has
  historically drifted on non-native KV dtypes; `VLLMServer.__init__`
  emits a `RuntimeWarning` when this combination is requested. Verify
  parity against a same-cell `kv_cache_dtype=auto` run before trusting
  the fp8 numbers.

## Sanity 2-point (run this before the full sweep)

Before spending a full sweep on any model (B × ctx × K), run the 4-cell
sanity to check whether the (B, ctx) K controller thesis is alive on
the target box:

```
python -m benchmarks.runner \
  --config benchmarks/configs/sanity_2point.yaml \
  --out results/sanity_2point
```

Fixed at `B=32`, sweeping `ctx ∈ {256, 2048}` × `K ∈ {0, 3}`. Verdict:

- **Survive** — at `ctx=2048`, `K=3` beats `K=0` on throughput or TPOT
  by a **≥ 10 %** margin (bootstrap 95 % CI clears 0). KV-read
  amortization keeps K > 0 optimal at long context — the mechanism this
  lab studies. Full sweep is worth running.
- **Falsify** — `K=3` margin at `ctx=2048` is **< 5 %** or CI includes
  0. Before declaring the mechanism dead, verify `drafter_loaded=True`
  and `spec_method_logged=='mtp'` in the recorded cells: a routing
  regression or drafter mis-alignment (Falsifier #6) can hide the
  mechanism before the mechanism itself is at fault.
- **Ambiguous** — margin between 5 % and 10 %. Re-run for a second seed
  before deciding.
- **Short-ctx cross-check** — at `ctx=256`, `K=3` should collapse
  toward the batch-only crossover behavior (small or no gain vs `K=0`
  at `B=32`). If `K=3` still wins at short ctx, the sanity is too
  easy — pick a batch closer to the batch-only crossover before drawing
  conclusions about the ctx-axis mechanism.

## V2 deployment gate

Sanity (V1) tells us the ctx-axis mechanism is alive. V2 tells us
whether the schedule the sweep produces is **safe to deploy in the
K=0 tier** — the batch regime where the controller demotes to K=0 for
production traffic. The gate answers one question:

> Does keeping the drafter resident but skipping speculation
> (`dsd_k0`, drafter loaded + K=0 per DSD schedule) cost measurably
> less throughput than not loading the drafter at all (`no_spec`)?
> If yes, the DSD schedule is production-safe. If no, the K=0 tier
> has a residual drafter-forward tax that the controller must handle
> another way (e.g. drafter unload/reload).

### Three arms

| arm              | `speculative-config`                                          | drafter VRAM | K per step        |
|------------------|---------------------------------------------------------------|--------------|-------------------|
| `dsd_k0`         | `method=mtp`, `dsd_schedule=[[1, B_max, 0]]`                  | loaded       | 0 (skip)          |
| `no_spec_v024`   | not emitted                                                   | not loaded   | —                 |
| `no_spec_v023`   | (reference only — prior no-spec throughput on vLLM 0.23)      | not loaded   | —                 |

`no_spec_v023` is the reference number carried over from the earlier
campaign (3413 tok/s at c=30 short-ctx). It guards against a
vLLM 0.24 upgrade tax: if `no_spec_v024` is materially lower than
`no_spec_v023` under the same workload, the drafter-tax verdict is
confounded by an engine regression and must be adjudicated separately
before the arm comparison is trusted.

### Grid

```
batch  : {128, 192, 256}    K=0 tier coverage (see § Sanity for lower B)
ctx    : {460, 970, 1990}   same three points as the dose-response probe
arm    : {dsd_k0, no_spec}  no_spec_v023 is a reference number, not a run
seed   : 3                  bootstrap-safe minimum
```

54 cells per campaign at 150 s per cell (120 s window + 30 s warmup)
plus per-cell vLLM startup (~2 min each — the runner brings the
engine up per cell today). Wall time ≈ 3.5–4 h. A single-engine
sweep (start once per arm, reuse across batch/ctx) is a P2.8-plus
optimization; V2 uses the current per-cell startup path so the
same code path stays under test.

### Verdict thresholds

Let `tax = (throughput[no_spec] - throughput[dsd_k0]) / throughput[no_spec]`
per (batch, ctx) cell, aggregated over the 3 seeds via bootstrap
(N = 10 000, 95 % CI).

- **Survive** — every (batch, ctx) cell's tax upper CI bound is
  **< 5 %**. DSD schedule ships. Proceed to P2.8 A (offline emitter)
  and P3.10 (results publication) in parallel; Layer B (runtime
  hot-swap) unblocks.
- **Ambiguous** — any cell falls in **5–10 %**. Re-run that cell for a
  second seed pool; if the tax stays in the band, treat as Falsify.
- **Falsify** — any cell's tax lower CI bound is **> 10 %**. The
  residual drafter-forward cost is not negligible in the K=0 tier.
  Controller must handle K=0 differently (candidates: drafter unload
  on sustained K=0 dwell, separate no-drafter engine instance,
  scheduler-side spec toggle). P2.8 B blocks; P2.8 A can still land
  as an offline evidence artifact.

### Confounds explicitly ruled out at V2

- **LMCache**: off by default (P1.6). V2 config carries no LMCache
  field; the LMCache × MTP shape (65) vs (128) bug (see
  `p1_vast_install_recipe.md`) is orthogonal to V2 and stays parked.
- **hit_rate**: 0.0 (APC, no prefix repeat). V2 measures the K=0 tier
  cost in the worst-case workload for the controller — no cache to
  amortize the drafter tax.
- **K > 0 cells**: not swept at V2. V1 sanity already established the
  ctx-axis mechanism on K=3 cells; V2 is specifically the K=0 tier.

### Running the gate

```
python -m benchmarks.runner \
  --config benchmarks/configs/v2_deployment_gate.yaml \
  --out results/v2_deployment_gate

python -m benchmarks.verdict \
  --results results/v2_deployment_gate/requests.parquet
```

`benchmarks.verdict` prints a per-cell tax with bootstrap 95 % CI
and an overall gate verdict (`SURVIVE`, `AMBIGUOUS`, `FALSIFY`).
Non-default thresholds via `--tax-survive-pct` / `--tax-falsify-pct`.

### Assertions during V2 (fail fast)

- `dsd_k0` arm cells must record `Δvllm:spec_decode_num_drafts_total == 0`
  over the measurement window. Non-zero drafts mean the DSD schedule
  did not clamp to K=0 for the requested batch — pipeline aborts and
  the run is discarded.
- `drafter_loaded` covariate must match arm: `True` for `dsd_k0`,
  `False` for `no_spec`. Enforced by the runner before the run starts.

## Branch topology

- `main` — merge target. All infra + all three configs live here.
- `dev-exaone4.5` — same-graph MTP head track. Owns the vLLM engine
  migration, `run_one` implementation, and LMCache wiring.
- `dev-gemma4` — external-drafter track. Adds Gemma 4 configs and
  the RTX 5090 32 GB budget.

## Falsifiers (do not discard before checking)

1. Prefill slack ≠ decode slack (phase mixing depends on chunked prefill)
2. Decode crosses into compute-bound regime at high B (small models / short ctx)
3. K ↑ inflates verify cost faster than acceptance rate amortizes
4. Hypothesis success window narrows under realistic hit-rate distributions
5. LMCache retrieval latency erases prefill-skip gain on short prompts.
   Isolated by comparing `apc` vs. `lmcache` arms at the same
   `(B, hit_rate, K)`.
6. External `*-it-assistant` drafter mis-alignment inflates rejection
   rate. **Order of investigation**: check `spec_method_logged` first
   — if the Gemma cells landed on `draft_model` instead of `mtp`, the
   fallback (silent target/drafter state divergence) explains a low
   acceptance rate before drafter alignment does. Only chase alignment
   after `_assert_spec_routing()` has confirmed method routing is
   correct.
7. Lever-size shortfall vs. mechanism absence. If the sanity 2-point
   falsifies, inspect the recorded `prefill_share` in the same cell:
   a low share means prefill compute is already small relative to
   decode, so `hit_rate` has little to remove and the mechanism can
   still be real. Re-run with larger `prompt_tokens` (or a
   longer-context workload) before declaring the hypothesis dead.

## Sweep output → vLLM 0.24 DSD schedule

The B×K table produced by the full sweep is directly reusable as the
input to vLLM 0.24 DSD's
`num_speculative_tokens_per_batch_size`. Extraction pattern:

```
per_B_optimal_K = argmax_K(throughput[B, hit_rate=<deployment>, K, arm=lmcache])
# → [[1,4,3],[5,16,2],[17,32,1],[33,512,0]]   (illustrative)
```

Phase 2 (post-sanity survival) is therefore *upgrade + config*, not
feature development. Phase 3 is the real feature contribution: DSD
today indexes only by B, but this sweep's `(hit_rate, K)` cross
section shows the optimal K is a function of `(B, prefill_share)`.
Adding a `prefill_share` dimension to the DSD lookup — with the sweep
table itself as the design evidence — is a first-of-its-kind
hit/prefill-aware K schedule. That is why `prefill_share` and
`kv_pool_tokens` are recorded on every cell.
