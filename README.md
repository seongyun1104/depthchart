# speculative-decoding-lab

Single-GPU experimentation lab for the hypothesis:

> LMCache prefill skip frees compute slack on the decode phase. MTP's verify cost is absorbed by that slack, so MTP speedup survives at high batch where prior work ("Performance or Illusion?", MLSys 2026) shows it collapses.

Scope: single H100/H200 only. Multi-instance, PD-disaggregation, llm-d are out.

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
speculative/   MTP / NEXTN / FastMTP / DFlash adapters; acceptance hooks
lmcache/       LMCache config; hit-rate-controlled workload gen
scheduler/     SGLang scheduler analysis + patch experiments
benchmarks/    sweep harness (B × hit_rate × K), metric collector
kv-transfer/   (out of P1 scope; placeholder for future)
papers/        REFERENCES.md — citations and gap mapping
```

## Stack

- Engine: SGLang (`--enable-lmcache`, EAGLE3 / NEXTN / DFlash)
- Model: EXAONE 4.5 33B FP8 (VLM, served text-only here)
- Spec head: EAGLE3-style draft head (per EXAONE MoE MTP ≡ EAGLE mapping)
- KV layer: LMCache (HBM → CPU → NVMe)
- Bench: built on LMBench (extended with spec-aware metrics)
- GPU: single H100 80GB (RunPod or equivalent)

## Sweep design

| variable     | values                                              |
|--------------|-----------------------------------------------------|
| batch size B | 1, 4, 16, 32, 64, 128                               |
| hit rate     | 0%, 30%, 60%, 90%                                   |
| spec K       | 0 (off), 1, 2, 3                                    |
| spec method  | eagle3 (EXAONE 4.5 draft head)                      |
| model        | `LGAI-EXAONE/EXAONE-4.5-33B-FP8` on single H100 80GB |

Metrics: TTFT, ITL, throughput, GPU compute util, memory bandwidth util,
MTP acceptance rate per (K, B), prefill-skip slack vs verify cost ratio.

## Falsifiers (do not discard before checking)

1. Prefill slack ≠ decode slack (phase mixing depends on chunked prefill)
2. Decode crosses into compute-bound regime at high B (small models / short ctx)
3. K ↑ inflates verify cost faster than acceptance rate amortizes
4. Hypothesis success window narrows under realistic hit-rate distributions
5. LMCache retrieval latency erases prefill-skip gain on short prompts
