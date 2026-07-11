# References

Pinned 2026-07-11. Load-bearing citations for the (B, ctx) K controller
thesis and the sweep/controller design. Full landscape lives in the
project memo `knowledge_speculative_lmcache_landscape_2026h1.md`.

## Theory (correctness of speculative decoding)

- **Fast Inference from Transformers via Speculative Decoding** —
  Leviathan et al., 2022. Original rejection-sampling proof that
  speculative decoding preserves the target model's output distribution.
- **Accelerating Large Language Model Decoding with Speculative
  Sampling** — Chen et al., DeepMind 2023. Independent formulation with
  the same distributional guarantee.

Both are the reason K > 0 does not sacrifice quality; the (B, ctx)
controller therefore optimizes throughput/TPOT under a fixed quality
budget, not under a quality tradeoff.

## Draft-head landscape (drafter zoo)

- **Better & Faster Large Language Models via Multi-token Prediction** —
  Gloeckle et al., 2024. Introduces the MTP-head training objective; the
  same-graph drafter architecture we drive on both Gemma 4 and EXAONE 4.5.
- **DeepSeek-V3** — 2024 technical report. Multi-token prediction at the
  target model scale; established MTP as a production speedup lever.
- **EAGLE / EAGLE-2 / EAGLE-3** — Li et al., 2024–2025. Feature-level
  draft with tree verification. vLLM and SGLang both route external
  drafters through the EAGLE path even when the method is nominally MTP.
- **FastMTP** — arXiv 2509.18362. Shared-weight MTP head with
  self-distillation; source of the per-position acceptance decay curves
  (K=1 80% / K=2 56% / K=3 36%) we compare Gemma 4 against.
- **DFlash** — arXiv 2602.06036, ICML 2026. Block diffusion draft +
  KV injection. Shipped in SGLang Spec V2 (PRs #22077, #23000).
  Secondary-pass sweep target after the EAGLE/MTP baselines are clean.
- **DSpark / DeepSpec** — DeepSeek 2026-06-27. Confidence-head +
  hardware-aware prefix scheduler on V4. Read as evidence that the
  engine-throughput axis has an existing implementation; the (B, ctx)
  axis this lab targets is orthogonal.

## Batch-axis dynamic K (shipped in production)

- **vLLM 0.24 Dynamic Speculative Decoding (DSD)** — the
  `num_speculative_tokens_per_batch_size` field on `SpeculativeConfig`.
  Schedule is a list of `[bs_lo, bs_hi, K]` tiers; the scheduler picks
  K from the tier that contains the per-step count of scheduled
  requests. DSD is the batch-axis prior art the (B, ctx) surface
  extends.
- **SGLang adaptive spec** — `--speculative-adaptive`, shipped default.
  Batch-tier candidate K sets (`{"1":[1,3,7], "8":[0,1,3],
  "32":[0,1], "64":[0]}`) plus in-tier acceptance-EMA selection.
  Adaptive spec params live in `sglang/srt/speculative/
  adaptive_spec_params.py`.

Both index only by batch size; neither uses context length as a
selection axis. That gap is the space this lab occupies.

## Related scheduler research (ctx-axis unclaimed)

None of these use context length as a first-class selection axis;
they are the closest points in the design space to (B, ctx).

- **DSDE** — KLD-divergence dispersion signal plus a straggler cap
  (per-sequence K override) to bound tail latency. Straggler-cap
  idea informs the deferred per-seq K extension in the K controller
  spec §7.
- **Nightjar** — Multi-armed bandit on/off decision, batch-size aware.
  A K∈{0, K_fixed} policy, no fine-grained K palette.
- **HeteroSpec** — Uses entropy of the target distribution as a
  "context complexity" signal to modulate K. Closest existing work
  to a ctx-aware K, but the signal is *token-level entropy*, not the
  KV-read amortization mechanism this lab measures.
- **BanditSpec** — MAB over K palette; batch-conditioned. No ctx axis.
- **AdaSpec** — arXiv 2503.05096, SoCC 2025. Adaptive on/off keyed on
  batch size and GPU utilization. No ctx axis.
- **LAPS-SD (Semi-Clairvoyant)** — arXiv 2505.17074, IJCAI 2025.
  Acceptance-rate-aware preemptive scheduler. No ctx axis.
- **TIDE** — Batch-conditioned performance model that predicts optimal
  K analytically from a service-time model. No empirical ctx axis.
- **SpecDec++** — RL-based K selection. Batch-conditioned. No ctx axis.

## Capacity track (separate from the DSD core)

Demoted per the P0 pivot: LMCache-serving lost throughput in load
(-63% GPU KV pool observation) and its role in the campaign narrowed
to KV-pool capacity, not the K-controller mechanism.

- **LMCache** — arXiv 2510.09665. Layered HBM → CPU → NVMe KV store.
  Official vLLM and SGLang integrations; up to 15× on cache-friendly
  workloads. The `_lmcache.yaml` overlay in this repo drives the
  capacity arm.
- **LMBench** — github.com/LMCache/LMBench. LMCache-org bench tool;
  no spec-aware metrics — an obvious follow-up OSS PR.

## Infra and background

- **Speculative Decoding: Performance or Illusion?** — arXiv 2601.11580,
  MLSys 2026. Systematic vLLM study showing target-verify cost dominates
  and MTP speedup collapses at high batch. The batch-only crossover this
  paper documents is the *K=0 side* of the (B, ctx) surface; the ctx
  axis is what keeps K > 0 alive past it.
- **vLLM** — `SpeculativeConfig` construction lives in
  `vllm/config/speculative.py`; the DSD scheduler indexes into
  `num_speculative_tokens_per_batch_size` via
  `vllm/v1/spec_decode/dynamic/utils.py::build_dynamic_sd_schedule_lookup`.
- **SGLang** — `python/sglang/srt/managers/scheduler.py` is the
  scheduler entry. Speculative routing (NEXTN / EAGLE / EAGLE3 / MTP)
  plus DFlash via Spec V2.
- **EXAONE 4.5 33B FP8** — `LGAI-EXAONE/EXAONE-4.5-33B-FP8`. Native MTP
  head (64 main + 1 MTP layer per model card). Track is deferred
  (drafter MAL ~1.5 leaves the AR signal too weak to distinguish
  drafter effects from covariate effects).
- **Gemma 4 QAT-FP8 (31B)** — `prithivMLmods/gemma-4-31B-it-qat-FP8`.
  QAT drafter `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`
  is the campaign winner (Cell A: AR 67.7%, 2,500 tok/s; +14% over the
  non-QAT Cell B drafter).

## Rejected / out-of-scope (recorded so we don't re-litigate)

- **TokenSpeed** — Preview-stage inference stack. K is a drafter-init
  argument baked into pre-allocated CUDA graphs; there is no runtime K
  switching. Gemma family is not in the supported models list.
  Judgment: not a substrate for this lab or a K-controller adapter
  target. Re-evaluate if Gemma support and dynamic K both land.
- **SPECTRE** — arXiv 2605.08151. Remote drafter, multi-tenant /
  multi-node. Out of P1 scope (single GPU).
- **QuantSpec** — OpenReview 7SHbJENgHX. Self-spec with hierarchical
  quantized KV. Orthogonal to the (B, ctx) axis; do not conflate.
- **Cacheback** — arXiv 2511.21699, EMNLP 2025. LRU n-gram cache as
  the draft. Different mechanism from prefix skip or KV-read
  amortization; not a comparable baseline.
