# Gate 2 result — SM90 FP8 candidate configs, isolated GEMM on H100 NVL (2026-11-18 KST)

Instance 61085155 (offer 60769714, BG, H100 NVL 94 GB, driver 595.91.07, billed $2.911/h), packet manifest
`03764c94…` (source `ec77f90`), created 23:52, box 23:55:34–00:10:51, evidence by one tar stream (345 files; the
per-file scp was stopped after 127), destroyed and absence confirmed 00:14:40. Vast credit 14.414 → 13.222.
Attempt 1 (61083526) stopped at identity on my loaded-.so check, no measurement: `evidence_attempt1_loaded_so/`.
Rules: `VERDICT_RULES_GATE2.md` (approved before the run). Tables: `RESULT_TABLES.txt` (`analyze_gate2.py`).

## A. Kernel identity — all 7 arms pass
W, 0, 1, 2, 3, 4, P: one CUTLASS kernel per call, (tile, cluster, schedule, swap AB) equal to the patch's dispatch table
at M 16–129 on gate_up and the N=2048 control; loaded `_C_stable_libtorch` hash = wheel `459044c5…` (W) or ours
`f2b60575…` (others).

## B. Correctness — all 7 arms pass the test bound (126/126 each)
Outside each arm's range every output is bitwise equal to arm 0 (no leak into other ranges). Inside the range:
C2, C3 and P bitwise equal to arm 0; swap-AB arms C1 and C4 differ (1/36 equal, max |Δ| 4.0 on outputs that pass
the bound). The recorded "real-input max_rel" (462, identical for every arm) is meaningless — a relative error over
near-zero references; my metric design, not a kernel signal.

## C. Timing (Σ of the 6 Gemma FP8 GEMMs, µs, mean of two mirrored passes)
| M | 0 (main) | C1 | C2 | C3 | C4 | P |
|---|---|---|---|---|---|---|
| 16 | 220.3 | | | | | 222.9 |
| 17 | 301.1 | | | | | **239.8** |
| 32 | 289.4 | | | | | **241.6** |
| 64 | 296.7 | 297.8 | 297.3 | 295.4 | 293.3 | 300.9 |
| 96 | 376.8 | 387.6 | 373.6 | 378.3 | 579.4 | 377.0 |
| 128 | 378.3 | 390.0 | 385.7 | 388.6 | 590.9 | 385.6 |

1. Build equivalence W vs 0: **pass** (−1.3 … +1.6 % at every M).
2. 64 | 65 step (M128 candidates): **no candidate lowers it.** C2 (+0.8 / −2.0 %) and C3 (−0.4 / −2.7 %) at M 96 / 128:
   no improvement detected under the pre-registered rule. C1 (−2.9 / −3.1 %) and C4 (−54 / −56 %): slower values
   observed, but **not attributable to the config** — control breach at M 129 and 192. The patch does not change those
   points (A: same kernel name for every arm), yet unexplained performance differences appeared there; a kernel name
   does not guarantee identical clocks, cache state, run order or buffers, so the breach is not dismissed as noise.
   ⇒ Verdict (BUILD_PLAN label): the 64 | 65 step is a property of main's config choice that **these four candidates
   do not remove**. Not concluded: that no SM90 config removes it.
3. 16 | 17 step, #66852's (16, 32] config (P): **lowers it** — M 17 / 24 / 32: −20.4 / −15.4 / −16.5 % in both passes,
   R = 0.76 / 0.69 / 0.69 — the share of the defined 16→17 increase of the isolated six-GEMM sum that is removed (not a
   share of server cost), controls held, outputs bitwise equal to main.
   ⇒ Verdict: evidence for the #66852 line (closed 11-12 "superseded", successor not named), not an independent finding.
4. Per shape (RESULT_TABLES.txt §4): P lowers every Gemma shape at M 17–32 (e.g. gate_up 101.3 → 81.5 µs at M 17).

## D. ncu — not measured
This host denies GPU performance counters (`ERR_NVGPUCTRPERM`), unlike gate 1's host; the probe saw 0 kernels and the
step was skipped by design.

## What can be said (scope)
| result | scope |
|---|---|
| C2 · C3 | no improvement detected under the pre-registered rule |
| C1 · C4 | slower values observed; not attributable to the config (control breach) |
| #66852 config | reproduces a GEMM-time improvement at M 17 · 24 · 32 on this H100 and harness |
| correctness · bitwise equality outside range | meets the test bound; supports isolation of the change to its range |
| server performance · bottleneck cause | not verified |

## Limits
Isolated GEMM, one H100 NVL host, two passes. No server step, no model, no other GPU. ν is arm 0's pass difference;
the M 129 / 192 breaches at points the patch does not change show between-arm differences can exceed it (rules and
verdicts unchanged after the run; the next experiment fixes its repeats, order and noise rule before it runs).
