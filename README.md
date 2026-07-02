# speculative-decoding-lab (dev-gemma4)

Single-GPU experimentation lab for the hypothesis:

> LMCache prefill skip frees compute slack on the decode phase. MTP's verify cost is absorbed by that slack, so MTP speedup survives at high batch where prior work ("Performance or Illusion?", MLSys 2026) shows it collapses.

**Branch scope.** This branch tracks the hypothesis on Gemma 4:

- `AxionML/Gemma-4-12B-NVFP4` on **RTX 5090 32 GB** (Blackwell `sm_120`, FP4-native)
- `prithivMLmods/gemma-4-31B-it-qat-FP8` on **H100 96 GB**

The parity branch `dev-exaone4.5` runs the same harness against
`LGAI-EXAONE/EXAONE-4.5-33B-FP8` with a same-graph native MTP head.
`main` is the merge target of the two.

Multi-instance, PD-disaggregation, and llm-d remain out of scope.

## Hypothesis

```
B ↑                       → MTP speedup → 0  (target verify dominates)
B ↑  +  LMCache hit ↑     → prefill skip → decode share ↑
                          → decode is memory-bound → compute slack
                          → MTP verify absorbed → MTP speedup survives
```

LMCache is expected to flatten MTP's batch-size scaling curve.

## Layout

```
speculative/   MTP / EAGLE / draft-model adapters; acceptance hooks
lmcache/       LMCache configs (cpu_only, cpu_disk, cpu_redis); hit-rate workload gen
scheduler/     vllm_runner.py (primary); sglang_runner.py kept as reference
benchmarks/    sweep harness (B × hit_rate × K); configs/gemma_4_*.yaml
kv-transfer/   (out of P1 scope; placeholder for future)
papers/        REFERENCES.md — citations and gap mapping
```

## Stack

- Engine: vLLM OpenAI-compatible server, launched via
  `scheduler/vllm_runner.py` with:
  - `--speculative-config '{"model":"google/gemma-4-<size>-it-assistant","num_speculative_tokens":K}'`
    (external draft model — Gemma 4 has no same-graph MTP head)
  - `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
  - `--enable-chunked-prefill`, `--max-num-batched-tokens`, TP=1
- Models & drafts:

  | main model | drafter | GPU target |
  |---|---|---|
  | `AxionML/Gemma-4-12B-NVFP4` | `google/gemma-4-12B-it-assistant` | RTX 5090 32 GB |
  | `prithivMLmods/gemma-4-31B-it-qat-FP8` | `google/gemma-4-31B-it-assistant` | H100 96 GB |

- Spec adapter: `speculative.adapter.for_gemma_4(k, draft_model)` emits a
  `SpecConfig` with `method="draft_model"`. Its
  `to_vllm_speculative_config()` returns
  `{"model": <drafter>, "num_speculative_tokens": K}` — no `method:`
  field, so vLLM routes it as an external draft rather than a
  same-graph head.
- KV layer: LMCache via `LMCACHE_CONFIG_FILE` env var
  (Phase 1 = HBM + CPU; Phase 2 = HBM + CPU + local NVMe)
- Bench: async harness in `benchmarks/runner.py`
  (streaming `/v1/chat/completions` + Prometheus `/metrics` polling)

## Fixed workload knobs

The harness pins these on every request to keep throughput comparable
across (B, hit_rate, K) cells:

- `ignore_eos: true` + `max_tokens: 256` — normalise decode length so
  cells are comparable at fixed output cost.
- `temperature: 0.0` — greedy decode; verify path determinism.
- `chat_template_kwargs.enable_thinking: false` — harmless for Gemma 4
  (no hybrid-reasoning template); kept for parity with `dev-exaone4.5`
  where it's load-bearing.

## Sweep design

| variable     | 12B NVFP4 / RTX 5090              | 31B QAT-FP8 / H100 96 GB          |
|--------------|-----------------------------------|-----------------------------------|
| batch size B | 1, 4, 16, 32                      | 1, 4, 16, 32, 64, 128             |
| hit rate     | 0%, 30%, 60%, 90%                 | 0%, 30%, 60%, 90%                 |
| spec K       | 0 (off), 1, 2, 3                  | 0 (off), 1, 2, 3                  |
| spec method  | draft_model                       | draft_model                       |
| drafter      | `google/gemma-4-12B-it-assistant` | `google/gemma-4-31B-it-assistant` |
| max_context  | 16384                             | 32768                             |

Metrics: TTFT, ITL, throughput, MTP acceptance rate per (K, B),
prefill-skip slack vs verify cost ratio.

## Memory budget

### RTX 5090 32 GB (Blackwell `sm_120`, NVFP4 native)

`gpu_memory_utilization = 0.85`:

```
32 GB × 0.85 = 27.2 GB  (weights + KV pool budget)
weights ≈ 12B × 0.5 B/param (FP4) = 6 GB
  → KV pool ≈ 27.2 - 6 = 21.2 GB
draft model (gemma-4-12B-it-assistant) resident: ~1-2 GB
reserve = 32 - 27.2 = 4.8 GB
```

Notes:
- Blackwell `sm_120` provides FP4 tensor cores natively; NVFP4 keeps
  matmul on-tensor-core without dequant overhead.
- If the drafter can't co-reside with the KV pool at hit ≥ 90%, drop
  `chunked_prefill_size` from 4096 → 2048, or lower
  `gpu_memory_utilization` to 0.80.

### H100 96 GB (Hopper `sm_90`, FP8 tensor cores)

`gpu_memory_utilization = 0.85`:

```
96 GB × 0.85 = 81.6 GB  (weights + KV pool budget)
weights ≈ 31B × 1 B/param (FP8) = 31 GB
  → KV pool ≈ 81.6 - 31 = 50.6 GB
draft model (gemma-4-31B-it-assistant) resident: ~4 GB (assumes BF16)
reserve = 96 - 81.6 = 14.4 GB
```

Notes:
- QAT-FP8 = the weights were quantization-aware-trained to FP8, which
  narrows the accuracy gap vs post-training FP8 on Gemma 4 31B.
- After the first sanity run, inspect free VRAM reported by vLLM at
  startup and tune `gpu_memory_utilization` up toward 0.90 if there's
  slack.

## Falsifiers (do not discard before checking)

1. Prefill slack ≠ decode slack (phase mixing depends on chunked prefill)
2. Decode crosses into compute-bound regime at high B (small models / short ctx)
3. K ↑ inflates verify cost faster than acceptance rate amortizes
4. Hypothesis success window narrows under realistic hit-rate distributions
5. LMCache retrieval latency erases prefill-skip gain on short prompts

For Gemma 4 specifically, add:

6. External `*-it-assistant` drafter mis-alignment inflates rejection
   rate — verify acceptance ≥ same-graph MTP baseline before drawing
   conclusions about hit-rate × K.
