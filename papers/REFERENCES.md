# References

Pinned 2026-06-30. See `~/.claude/.../memory/knowledge_speculative_lmcache_landscape_2026h1.md`
for the full landscape — this file holds the subset that is load-bearing
for the hypothesis or for the sweep design.

## Core (load-bearing for hypothesis)

- **Speculative Decoding: Performance or Illusion?** — arXiv 2601.11580.
  First systematic vLLM-based study showing target-verify cost dominates and
  MTP speedup collapses at high batch. This is the curve we expect to
  flatten by adding LMCache.

- **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**
  — arXiv 2510.09665. Layered HBM → CPU → NVMe. Official vLLM and SGLang
  integration. Up to 15× throughput on cache-friendly workloads. Source of
  the prefill-skip mechanism we rely on.

- **FastMTP** — arXiv 2509.18362, Tencent-BAC. Shared-weight MTP head with
  self-distillation; K=3 avg 2.03×; per-step acceptance K=1 80% / K=2 56% /
  K=3 36%. We borrow these numbers as the baseline acceptance curve and
  measure whether LMCache shifts the curve under load.

## Spec methods in scope

- **DFlash** — arXiv 2602.06036, ICML 2026. Block diffusion + KV injection.
  Shipped in SGLang Spec V2 (PRs #22077, #23000). Second-pass sweep target
  after EAGLE3 baseline is clean.
- **DSpark / DeepSpec** — DeepSeek 2026-06-27. Confidence head + hardware-
  aware prefix scheduler on V4. We do not chase DSpark; we read it as
  evidence that the engine-throughput axis already has an implementation
  and our axis (LMCache hit rate) is orthogonal.

## Scheduler comparisons (so we don't reinvent)

- **AdaSpec** — arXiv 2503.05096, SoCC'25. Adaptive on/off via batch size +
  GPU util. Up to 66% over SOTA. Does not consider cache hit.
- **LAPS-SD (Semi-Clairvoyant)** — arXiv 2505.17074, IJCAI 2025.
  Acceptance-rate-aware preemptive scheduler. 39% latency cut. Not cache
  aware.

## Adjacent (read but not in scope)

- **SPECTRE** — arXiv 2605.08151. Remote drafter. Multi-tenant, multi-node.
  Out of P1 scope (single GPU).
- **QuantSpec** — OpenReview 7SHbJENgHX. Self-spec with hierarchical
  quantized KV. Orthogonal to our tiering axis.
- **Cacheback** — arXiv 2511.21699, EMNLP 2025. LRU n-gram cache as draft.
  Mechanism different from LMCache prefix skip; do not conflate.

## Infra

- **LMBench** — github.com/LMCache/LMBench. LMCache org's official bench
  tool. Currently no spec-aware metrics → an extension is a natural OSS PR.
- **SGLang** — `python/sglang/srt/managers/scheduler.py` is the scheduler
  entry. `--enable-lmcache`, NEXTN / EAGLE / EAGLE3, DFlash via Spec V2.
- **EXAONE 4.5 33B FP8** — `LGAI-EXAONE/EXAONE-4.5-33B-FP8`. VLM, used
  text-only here. Native MTP head built into the model (64 main +
  1 MTP layer per the model card). vLLM serves it via `--speculative_config
  '{"method":"mtp",...}'`; SGLang can drive the same head via the `MTP`
  algorithm flag (preferred) or via `EAGLE` (model-card fallback). We
  sweep both methods.
