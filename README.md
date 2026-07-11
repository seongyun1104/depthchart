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
across (B, hit_rate, K) cells and across models:

- `chat_template_kwargs.enable_thinking: false` — EXAONE 4.5's hybrid
  reasoning template inflates completion length ~21% when thinking is
  on, which contaminates the throughput signal. Harmless for Gemma 4.
- `ignore_eos: true` + `max_tokens: 256` — normalise decode length so
  cells are comparable at fixed output cost.
- `temperature: 0.0` — greedy decode; verify path determinism.

## Sweep design

Each track uses its own YAML. Cell counts are trimmed to fit VRAM.

| variable     | EXAONE 4.5 33B FP8 / H100         | Gemma 4 12B NVFP4 / RTX 5090      | Gemma 4 31B QAT-FP8 / H100        |
|--------------|-----------------------------------|-----------------------------------|-----------------------------------|
| batch size B | 1, 4, 16, 32, 64, 128             | 1, 4, 16, 32                      | 1, 4, 16, 32, 64, 128             |
| hit rate     | 0%, 30%, 60%, 90%                 | 0%, 30%, 60%, 90%                 | 0%, 30%, 60%, 90%                 |
| hit source   | `apc`, `lmcache`                  | `apc`, `lmcache`                  | `apc`, `lmcache`                  |
| spec K       | 0 (ref), 1, 2, 3                  | 0 (ref), 1, 2, 3                  | 0 (ref), 1, 2, 3                  |
| spec method  | `mtp` (same-graph)                | `mtp` (external drafter)          | `mtp` (external drafter)          |
| drafter      | (native)                          | `google/gemma-4-12B-it-assistant` | `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` |
| max_context  | 32768                             | 16384                             | 32768                             |

Metrics: TTFT, ITL, throughput, MTP acceptance rate per (K, B),
prefill-skip slack vs verify cost ratio. Covariates recorded per cell:
`kv_pool_tokens` (parsed from the vLLM `GPU KV cache size: N tokens`
line), `prefill_share` (Prometheus prompt/generation token delta over
the measurement window), `drafter_loaded` (True for K ≥ 1),
`hit_source`, `enable_prefix_caching`, `enable_lmcache`.

### Hit-source arms (why `hit_source` is an axis, not a knob)

`enable_prefix_caching` and LMCache both produce prefix-cache hits, and
if both are on the workload's repeated prefix is served from HBM
before the LMCache retrieval path is even exercised — the `hit_rate`
axis then measures native APC, not LMCache, and Falsifier #5 becomes
un-testable. The sweep splits hit provenance explicitly:

- **Arm `apc`** — native `--enable-prefix-caching` on, LMCache off. Hits
  come from HBM. Measures the *mechanism upper bound*: how much decode
  slack does prefill-skip create when the transfer cost is zero.
- **Arm `lmcache`** — native prefix caching off
  (`--no-enable-prefix-caching`), LMCache on with the configured
  `cpu_only` / `cpu_disk` backend. Hits come from CPU/NVMe. The gap
  vs. the `apc` arm at the same `(B, hit_rate, K)` cell is the direct
  cost of LMCache retrieval — the number Falsifier #5 needs.

The sanity 2-point therefore expands to 4 cells (2 hit rates × 2 arms
at fixed `B=32, K=2`).

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

Before spending a full sweep on any model (B × hit × K × arm = 6×4×4×2
= 192 cells at the widest), run the 4-cell sanity to check the
hypothesis is even alive on the target box:

```
python -m benchmarks.runner \
  --config benchmarks/configs/sanity_2point.yaml \
  --out results/sanity_2point
```

Fixed at `B=32, K=2`, sweeping `hit ∈ {0.0, 0.9}` × `hit_source ∈
{apc, lmcache}`. Verdict is read off the `apc` arm first (mechanism
upper bound), then the `lmcache` arm (system upper bound after
retrieval cost):

- **Survive (mechanism)** — on the `apc` arm, `hit=0.9` beats
  `hit=0.0` on throughput or ITL by a **≥ 10 %** margin (bootstrap
  95 % CI clears 0). Prefill compute is genuinely convertible into
  decode slack that MTP verify absorbs. Full sweep is worth running.
- **Falsify (mechanism)** — `apc` arm margin **< 5 %** or CI includes
  0. Before declaring the hypothesis dead, check `prefill_share` in
  the `hit=0.0, apc` cell (Falsifier #7): if it is already low, the
  lever is small, not absent — re-run with larger `prompt_tokens`.
  Otherwise Falsifier #2 (decode compute-bound) wins.
- **Ambiguous (mechanism)** — margin between 5 % and 10 %. Re-run for
  a second seed before deciding.
- **System gap** — regardless of the mechanism verdict, compare
  `apc` vs `lmcache` at `hit=0.9`. The delta is the LMCache retrieval
  cost (Falsifier #5). If the `lmcache` arm loses more than ~30 % of
  the `apc` arm's gain over `hit=0.0`, the "LMCache × MTP" claim is
  compromised even when the mechanism itself is real.

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
