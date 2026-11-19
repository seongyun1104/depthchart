# Gate 2 verdict rules (fixed 2026-11-17 before any gate-2 GPU run; thresholds approved by the user 23:4x KST)

Arms: W = wheel's own `_C_stable_libtorch` (upstream build, env ignored) · 0 = our build, no selector ·
1–4 = `VLLM_TEST_SM90_FP8_M128_CAND` (BUILD_PLAN.md table) · P = `VLLM_TEST_SM90_FP8_PR66852=1`.
Everything below is the isolated GEMM on H100 (upstream `benchmark_fp8_gemm.py`, `fp8-channel-w-token-a-noquant`,
`do_bench_cudagraph` median). No server, no model.

## Step A — kernel identity (gate; stops the box)
Every arm, gate_up 43008×5376 and control 2048×5376, M ∈ {16, 17, 32, 33, 64, 65, 96, 128, 129}: exactly one CUTLASS
kernel, and its (tile, cluster, schedule, swap AB) equals the patch's dispatch table (`gate2_check.expected`), and the
loaded `_C_stable_libtorch` hash equals the arm's .so. One mismatch → STOPPED, no timing, no verdict.

## Step B — correctness (per arm; a failing arm is out, the bound is not relaxed afterwards)
`test_cutlass_scaled_mm`'s helper inputs and bound (rtol 0.5, atol 0.15) vs FP32 baseline, 6 Gemma shapes + N=2048
control × the nine M × bias {off, on}. Recorded, not gated: bitwise equality and max-abs vs arm 0 (expected equal
outside each arm's range), harness-style real-valued inputs vs FP32 product.

## Step C — timing
t(arm, shape, M) = 2·M·N·K / TFLOP/s. S(arm, M) = Σ over the 6 Gemma shapes (the per-step FP8 GEMM sum, as run 2).
Two passes, order W 0 1 2 3 4 P then reversed. Noise ν(M) = |S_p1 − S_p2| of arm 0; value = mean of the two passes.

1. Build equivalence: |S_W − S_0| / S_0 ≤ max(2ν, 2%) at every M. If not, candidate verdicts are stated relative to
   our build (arm 0) only, never to the wheel.
2. Candidate X ∈ {1..4} lowers the 64 | 65 step if, at M = 96 **and** 128, S_0 − S_X > max(2ν, 3%) of S_0 in **both**
   passes; reported with R = (S_0(M) − S_X(M)) / (S_0(M) − S_0(64)) (fraction of the step removed).
   Controls must hold: M ≤ 64, M ≥ 129, and the N = 2048 shape at every M change by ≤ max(2ν, 2%).
   A control breach makes that arm "not attributable" whatever its in-range result.
3. P lowers the 16 | 17 step if, at M = 17, 24 **and** 32, S_0 − S_P > max(2ν, 3%) in both passes; controls M = 16,
   33–64, ≥ 65 and N = 2048 at M ≥ 33 hold as in 2.
4. Per-shape tables are reported for every arm; the sum decides.

## Verdict labels (BUILD_PLAN.md, user)
- No arm passes 2 → step is a phenomenon of main's config choice that these candidates do not remove.
- An arm passes 2 and B → kernel-fix candidate (isolated GEMM only; server effect untested).
- P passes 3 and B → evidence for the #66852 line (16 | 17), not an independent finding.
- Isolated GEMM and server disagree later → cause open.
Not concluded by gate 2: server step time, other GPUs, other M ranges, tolerance beyond the test bound.

## Step D — ncu (descriptive only)
Probe first (one device_kernel row) → per arm 0–4, gate_up at M 64, 96, 128: DRAM bytes read, duration, grid, block,
cluster; tensor-pipe (GMMA) active, DRAM/SM throughput, warps active. No verdict from ncu.
