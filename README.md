# speculative-decoding-lab

Single-GPU experimentation lab for the hypothesis:

> LMCache prefill skip frees compute slack on the decode phase. MTP's verify cost is absorbed by that slack, so MTP speedup survives at high batch where prior work ("Performance or Illusion?", MLSys 2026) shows it collapses.

Scope: single H100 96GB only. Multi-instance, PD-disaggregation, llm-d are out.

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
speculative/   MTP / NEXTN / EAGLE / DFlash adapters; acceptance hooks
lmcache/       LMCache configs (cpu_only, cpu_disk); hit-rate workload gen
scheduler/     SGLang scheduler analysis + patch experiments
benchmarks/    sweep harness (B × hit_rate × K), metric collector
kv-transfer/   (out of P1 scope; placeholder for future)
papers/        REFERENCES.md — citations and gap mapping
```

## Stack

- Engine: SGLang (`--enable-lmcache`, native MTP head served via `MTP` algo;
  EAGLE / NEXTN kept as fallback sweep options)
- Model: `LGAI-EXAONE/EXAONE-4.5-33B-FP8` (VLM, served text-only)
- Spec head: EXAONE 4.5 has a native MTP head (1 MTP layer on top of 64
  main layers). No separate draft checkpoint needed.
- KV layer: LMCache via `LMCACHE_CONFIG_FILE` env var
  (Phase 1 = HBM + CPU; Phase 2 = HBM + CPU + local NVMe)
- Bench: built on LMBench (extended with spec-aware metrics)
- GPU: single H100 96 GB (TP=1)

## Sweep design

| variable     | values                                                |
|--------------|-------------------------------------------------------|
| batch size B | 1, 4, 16, 32, 64, 128                                 |
| hit rate     | 0%, 30%, 60%, 90%                                     |
| spec K       | 0 (off), 1, 2, 3                                      |
| spec method  | mtp (default) / eagle (fallback per EXAONE card)      |
| eagle_topk   | 1                                                     |
| draft_tokens | 4                                                     |
| model        | `LGAI-EXAONE/EXAONE-4.5-33B-FP8` on single H100 96 GB |

Metrics: TTFT, ITL, throughput, GPU compute util, memory bandwidth util,
MTP acceptance rate per (K, B), prefill-skip slack vs verify cost ratio.

## Memory budget (mem_fraction_static = 0.85)

```
mem_fraction_static = (weights + KV pool) / GPU capacity
SGLang rule of thumb: leave 5-8 GB for activations + CUDA graphs.

96 GB × 0.85 = 81.6 GB  (weights + KV pool budget)
weights ≈ 33 GB  →  KV pool ≈ 48.6 GB
reserve = 96 - 81.6 = 14.4 GB
```

The 14.4 GB reserve is conservative (rule of thumb is 5-8 GB) to absorb
MTP draft-tree activation spikes and LMCache transfer buffers. After the
first sanity run, inspect `available_gpu_mem` in the SGLang log and tune
up toward 0.90 if there is slack.

## Falsifiers (do not discard before checking)

1. Prefill slack ≠ decode slack (phase mixing depends on chunked prefill)
2. Decode crosses into compute-bound regime at high B (small models / short ctx)
3. K ↑ inflates verify cost faster than acceptance rate amortizes
4. Hypothesis success window narrows under realistic hit-rate distributions
5. LMCache retrieval latency erases prefill-skip gain on short prompts
