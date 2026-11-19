# Same K, different GEMM path: server step cost on Gemma-4 31B FP8 (H100 NVL, MRv2)

**Finding.** At the same static K, the server step cost changed with which FP8 GEMM path the step's input shape
selected. With K1 at batch 16, every target step has M = 32 tokens. That M falls in the CUTLASS SM90 FP8 bucket
that main dispatches to a slower config. Applying the (16, 32] branch of vllm-project/vllm#66852 lowered that step by
**7.6 % at 1k context and 6.5 % at 4k**. Eight control cells, which have M outside the branch, stayed within ±0.3 %.

This is a reproduction of the #66852 line at server level, credited to that PR. It is not an independent finding.
No new kernel PR is proposed from this work.

Setup for everything below: `prithivMLmods/gemma-4-31B-it-qat-FP8` target with the
`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` drafter, vLLM Model Runner V2, `TRITON_ATTN`, one H100 NVL
(94 GB) per run.

## 1. K cost map (server, measured) — [`COST_MAP.md`](COST_MAP.md)

The map covers 10 (batch, context) shapes × {OFF, Z, K1, K3, K5}, with two launches each, under a pre-registered
comparison rule ([`STAGE1_R2_PREREG.md`](../costmap_20261107/STAGE1_R2_PREREG.md), result
[`RESULT_STAGE1_R2_20261111.md`](../costmap_20261107/run_20261111r2/RESULT_STAGE1_R2_20261111.md)).

- Step period grows with K at every shape. Relative to OFF, K3 is +11.6 to +42.8 % and K5 is +19.2 to +70.2 %, and
  the increase grows with batch and context.
- K1 costs about the same as Z (drafter loaded, K = 0) at 6 of 10 shapes. The exception is batch 16, where K1 is
  +10.4 to +10.6 % over Z. That is the cell whose target M moves from 16 to 32.
- Delivered ms per token depends on acceptance. Acceptance here is 0.65–1.0 because the prompts are random tokens
  with `ignore_eos` and 640 output tokens. Delivered ms per token is a cost-map axis, not a speedup estimate for real
  traffic.

The last table in `COST_MAP.md` marks which M = 32 cells were **measured** with the #66852 config (K1 · b16) and
which are **projections**. The projections assume the server realises r = 0.76 of the isolated GEMM saving, a value
taken from one observation. K3 · b8 is also M = 32, but its step has a different composition, so it stays a
projection until it is measured.

## 2. FP8 GEMM candidates for the 64 | 65 step — negative result

Isolated GEMM timing of the six Gemma FP8 shapes shows steps at 16 | 17, 64 | 65 and 128 | 129
([`RESULT_GEMM_RUN2_20261117.md`](../kernel_probe_20261117/run_gemm2/RESULT_GEMM_RUN2_20261117.md)). Gate 2 tried
four SM90 configs for the M 65–128 bucket, built into one `.so` and selected by environment variable
([patch](../kernel_probe_20261117/gate2/sm90_fp8_candidates.patch), rules
[`VERDICT_RULES_GATE2.md`](../kernel_probe_20261117/gate2/run_g2/VERDICT_RULES_GATE2.md), result
[`RESULT_GATE2_20261118.md`](../kernel_probe_20261117/gate2/run_g2/RESULT_GATE2_20261118.md)).

- **No candidate lowered the 64 | 65 step.**
  - C2 and C3: no improvement was detected under the pre-registered rule.
  - C1 and C4 measured slower, but the slowdown cannot be attributed to the config. Controls at M 129 and 192 also
    moved, and the patch does not change those points: every arm ran the same kernel there.
  - The cause of that control drift is unexplained. It is reported, not dismissed as noise.
- **Scope:** this shows that these four candidates do not remove the step. It does not show that no SM90 config can.
- In the same run, #66852's (16, 32] config lowered the isolated six-GEMM sum by 15–20 % at M 17 / 24 / 32. Its
  outputs were bitwise equal to main, and its controls held.

## 3. Server comparison: #66852's (16, 32] branch at M = 32 — `lowers`

The comparison was pre-registered as tag `server-cmp-66852-prereg-20261119`
([rules](../kernel_probe_20261117/server_cmp/PREREG_SERVER_66852_DRAFT_20261119.md),
[scope of the patched branch](../kernel_probe_20261117/server_cmp/P_BRANCH_SCOPE_20261119.md)). Both arms used the
same `.so`, so the only difference was one selector environment variable. The six launches ran in the order
0 P P 0 0 P. Each launch's server processes passed an identity check on the mapped file and the environment.
Result: [`RESULT_SERVER_66852_20261119.md`](../costmap_20261107/server_cmp/run_srv/RESULT_SERVER_66852_20261119.md).

| cell | M | arm 0 step (ms) | arm P step (ms) | P vs 0 | saved |
|---|---:|---|---|---:|---:|
| K1 · b16 · 1k | 32 | 24.70 / 24.69 / 24.68 | 22.83 / 22.86 / 22.78 | **−7.6 %** | 1.87 ms |
| K1 · b16 · 4k | 32 | 26.38 / 26.51 / 26.50 | 24.69 / 24.90 / 24.65 | **−6.5 %** | 1.71 ms |
| 8 controls (M 2, 16, 64) | | | | within ±0.3 % | |

- In both test cells, all three P launches were below all three arm-0 launches, by more than twice the 3 % threshold.
- Acceptance was equal across arms within 0.006, and every steady step ran at the planned M.
- The saving is 0.84–0.91 of the isolated-GEMM estimate (2.05 ms), and 0.67–0.80 of r2's K1 − Z step increase at
  b16.
- Arm 0 reproduced r2's K1 · b16 step (24.69 vs 24.73 ms at 1k, 26.46 vs 26.44 ms at 4k) on a different host and
  day.

## What is not claimed

- No throughput or end-to-end speedup, and no result for other M values, GPUs or models.
- Nothing about #66852's M ≤ 16 branches, or about whether current upstream main sees the same effect. The
  dispatch file on main has no equivalent change.
- No optimal K or policy ranking.
- K3 · b8 with the config is a projection, not a measurement.
- The 64 | 65 result covers four candidates only.
- The original question of this lab, whether a context-length axis in the K schedule beats a well-tuned batch-only
  schedule, is **not answered here**. See [`EVIDENCE_LEDGER.md`](../../EVIDENCE_LEDGER.md).
  - On the same r2 data, **no K-rank reversal across context was found for this workload and the K range measured**
    (Z, K1, K3, K5). The ranking used delivered ms per token, and a reversal had to be resolved by the registered
    two-launch rule at both contexts of a batch
    ([rules](../k_rank_r2_20261119/RULES.md), committed before computing; [result](../k_rank_r2_20261119/RESULT.md)).
  - At batch 8, K5's margin over K3 narrows with context (log ratio 0.34 → 0.24 → 0.06). This is an observation
    only: the order does not reverse inside the grid.

## Raw data and how to reproduce

Every run directory has `evidence/costmap_out/` and an `EVIDENCE_SHA256SUMS`. The offline commands below regenerate
the committed outputs from the committed evidence. They need Python 3 and NumPy, and no GPU. Run them from
`mrv2_ctx_20261105/`.

```bash
# r2 cost map report: the analysis script as registered (public_20261119/repro_r2_tag/) reproduces the committed
# report byte for byte; the later version in costmap_20261107/ adds one field per cell (k_selection)
python public_20261119/repro_r2_tag/analyze_costmap.py costmap_20261107/run_20261111r2/evidence/costmap_out \
  > /tmp/r2.json && cmp /tmp/r2.json costmap_20261107/run_20261111r2/costmap_report_local.json
python costmap_20261107/run_20261111r2/analysis/export_csv.py costmap_20261107/run_20261111r2   # cells.csv, pairs.csv, launches.csv

# gate 2 (isolated FP8 GEMM, 7 arms)
python kernel_probe_20261117/gate2/run_g2/analyze_gate2.py kernel_probe_20261117/gate2/run_g2/evidence/costmap_out \
  | cmp - kernel_probe_20261117/gate2/run_g2/RESULT_TABLES.txt

# server comparison verdict (JSON-equal to the committed compare_arms.json)
python costmap_20261107/compare_arms.py costmap_20261107/server_cmp/run_srv/evidence/costmap_out > /tmp/arms.json

# M = 32 projection, this cost map, and the K-rank check
python kernel_probe_20261117/server_cmp/m32_projection.py kernel_probe_20261117/gate2/run_g2/evidence/costmap_out \
  costmap_20261107/run_20261111r2/analysis/cells.csv | cmp - kernel_probe_20261117/server_cmp/m32_projection.json
python public_20261119/build_cost_map.py | cmp - public_20261119/COST_MAP.md
python k_rank_r2_20261119/k_rank.py | cmp - k_rank_r2_20261119/RESULT.json
```
The GPU box scripts are included for reference: `costmap_20261107/stage1/run_stage1_box.sh`,
`kernel_probe_20261117/gate2/run_g2/run_gate2_box.sh` and `costmap_20261107/server_cmp/run_server_cmp_box.sh`. Each
launch's server command and plan are in its evidence (`cmd.json`, `plan.json`). The packet builders and the full lab
history are not part of this branch, so a GPU rerun needs those to be rebuilt.
