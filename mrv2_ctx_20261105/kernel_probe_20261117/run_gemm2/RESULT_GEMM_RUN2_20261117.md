# SM90 FP8 GEMM M-bucket check — run 2 result (2026-11-17)

Instance 61006794 (offer 60769722, H100 NVL 95,830 MiB, driver 595.91.07, $3.104/h), packet `61e06c7` / manifest
`04d18981…`, created 11:56 KST, ALL_DONE 12:07:56, collected 12:11:02, destroyed and absence confirmed 12:11:15.
Engine binaries = precompiled wheel of main `7ecc83a9` (dispatch file identical on main `b5a2d209`). Harness = that
commit's `benchmark_fp8_gemm.py` + `harness.patch`; two full passes (rep0, rep1). Analysis: `analyze_gemm.py`,
`analysis_rep{0,1}.json`. Run 1 (61005199) stopped on the harness's folder order, no data.

## Verdict under VERDICT_RULES.md: steps confirmed; config cause not established (partial)
- Boundaries (Gemma per-step GEMM sum, both passes, beyond the pass-to-pass gap): 16|17 +3.50 / +3.85 ms,
  64|65 +3.28 / +3.20 ms, 128|129 +6.01 / +5.73 ms; 32|33 no step (−0.10 / −0.05). Flat inside buckets.
- Transitions vs the server step jump (size correspondence only, not a causal share; 145% at 16→32 shows the
  isolated GEMM sum and the server step are not interchangeable): 16→32 +3.24 / +3.49 ms vs 2.33 (145%, the GEMM sum exceeds the server jump);
  64→96 +4.03 / +4.19 vs 5.30 (78%); 64→128 +5.69 / +5.74 vs 7.10 (81%); 128→192 +5.93 / +6.01 vs 11.73 (51%).
  Inside-bucket controls 8→16 −0.20 / −0.30, 32→64 −0.49 / −0.24 (not beyond the gap).
- The pre-registered config proxy failed: 2·t(16)/t(32) is 1.07–1.77 on every shape (two M ≤ 16 calls are slower than
  one M = 32 call). The proxy was ill-chosen — in this memory-bound range two calls read the weights twice — but the
  rule stands as written: no "config cause" claim from this run.
- Descriptive, not part of the verdict: effective bandwidth (weight bytes ÷ time; DRAM traffic not measured) falls by bucket on the large shapes, e.g. Gemma
  gate_up 3.03 TB/s at M 16 → 2.15–2.24 at 17–64 → 1.88 at 65, 1.51 at 128 → 1.24 at 129 (Qwen3 gate_up and Gemma
  qkv / down alike), 25–60% lower after each boundary. Whether DRAM reads are the same is not established.
- Same boundaries in all three models (per-shape ratios, both passes): 16→17 1.16–2.01, 64→65 1.15–1.83, 128→129
  1.04–2.24; largest on the small-N/K shapes (o_proj, Llama qkv). Sliding o_proj (≈ 44 MB) may sit in L2 here.
- Cost falling with size (step sum, both passes beyond the gap): 2→4 (−0.65 / −0.69 ms) and 40→48 (−0.55 / −0.61 ms),
  4–6%. A running max (adaptive verification's build_table) would overwrite those by 6% / 9%. Small; the server step
  data showed no repeated drop. Per-shape drops on the smallest Llama shapes (6–7 µs) are frequent and treated as
  timing granularity.

## Mapping to prior work
- 16|17 (b16·K1): #66852 (closed unmerged) retuned exactly M ≤ 32 for N > 1280, K ≥ 1024 — not independent.
- 64|65 and 64→96 / 64→128 with N > 2048: outside #66852 (M ≤ 32), #66126 (SM90 part N ≤ 2048), #64521 / #65297
  (M > 128). The step is real and its size corresponds to ~80% of the server jumps (not a causal share); whether another config lowers it is untested.
- 128|129: overlaps #64521 / #65297's range.

## Next gates (none run)
1. Kernel identity by profiler (separate from timing) at M 16, 17, 64, 65, 128, 129.
2. Candidate configs for 65 ≤ M ≤ 128, N > 2048 (and #66852's for M ≤ 32) built off the GPU box; same harness, A/A
   and order-swapped A/B; correctness against an FP32 reference with tolerances fixed in advance.
3. Server main vs patch on the cells that move (b16 K3→K5, b32 K1→K3).

(Wording corrected 2026-11-17 after user review: correspondence vs causal share; effective vs measured bandwidth.)
