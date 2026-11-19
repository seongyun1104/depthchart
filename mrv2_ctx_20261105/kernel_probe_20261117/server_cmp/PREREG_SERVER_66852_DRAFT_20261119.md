# Server comparison — main config vs #66852 (16, 32] config — pre-registration DRAFT (2026-11-19, no GPU)

Status: **DRAFT, design approved.** User 2026-11-19 ~09:2x KST: K1 only (§3), order 0 P P 0 0 P (§4), §7 numbers as
drafted. The "(proposal)" marks below are kept as written; they are now fixed. Registration (tag + packet) follows §9.
Inputs: `gate2/run_g2/RESULT_GATE2_20261118.md`, `gate2/run_g2/LINEAGE_AND_SERVER_USE_20261118.md`,
`costmap_20261107/run_20261111r2/` (r2 cells, `B16_K1_TRACE_CHECK_20261111.md`).

Correction 2026-11-19 (before approval took effect on code): §3 said plan seed 20261107; r2 ran `PLAN_SEED=20261109`
(`run_stage1_box.sh`, r2 `launches.jsonl`). The plan uses **20261109** so the cell order equals r2's (prompt seeds are per shape, 20000 + i, independent of the plan seed).

## 1. Question
Does the #66852 (16, 32] SM90 FP8 config, which lowered the isolated six-GEMM sum at M 17 / 24 / 32 by 15–20 % in gate 2,
lower the **server's decode step period** in a cell whose target forward runs at M = 32, without changing cells that
run at other M?
Not asked: 64 | 65 step, other GPUs, other models, real traffic, speedup. A positive answer is a reproduction of the
#66852 line at server level, not an independent finding (decision (120): no kernel PR under our name).

## 2. Arms (the only difference = one environment variable)
**Scope of P (checked 2026-11-19, `P_BRANCH_SCOPE_20261119.md`)**: the build carries only #66852's (16, 32] branch
(64×32×512 for M ≤ 32, N > 1280, K ≥ 1024). #66852's M ≤ 8 / M ≤ 16 K-512 branches are not in the patch or the .so
(no 64×8×512 / 64×16×512 kernel), and gate 2 saw P equal to arm 0 at M 16. So b1 (M 2) and b8 (M 16) are valid
controls, and every statement names "#66852's (16, 32] config", never "#66852" as a whole.

| arm | `_C_stable_libtorch.abi3.so` | env |
|---|---|---|
| 0 | ours, sha256 `f2b60575…061db` (main + `sm90_fp8_candidates.patch`) | `VLLM_TEST_SM90_FP8_PR66852` unset |
| P | same file | `VLLM_TEST_SM90_FP8_PR66852=1` |

- Both arms load the same .so, so build differences cancel. The wheel's own .so is not an arm; gate 2 found W vs 0
  equal within ±1.6 % (isolated GEMM only), and r2's K1 values (wheel .so, 11-11, other host) are printed beside arm 0
  as a reference, never pooled or tested.
- The selector is read once per process (`static const int p66852`) and applies only when 16 < m ≤ 32, n > 1280,
  k ≥ 1024. The env var is set before `vllm serve`; CUDA graphs are captured after it is read.
- Engine as r2 / P1: tree `eeb37ef757` (patch `bs_ctx_bd09764c45.patch` on 7ecc83a9, Python only, 12 files, no csrc),
  base wheel 7ecc83a9, same models and pins, `costmap_cmd.SERVE_FLAGS` unchanged. The .so goes to
  `/workspace/vllm/vllm/_C_stable_libtorch.abi3.so` (same path in gate 2 and in `preflight_mrv2.sh`).

## 3. Cells (proposal: K1 only, the registered stage-1 plan unchanged)
Config **K1** in every launch, plan `stage1` (seed 20261109 as r2, same 10 shapes, warm-up + 2 sentinels), so arm 0
is directly comparable with r2's K1 launches. Target M = BS × 2:

| role | shapes | target M | expected |
|---|---|---|---|
| **test** | b16·1k, b16·4k | 32 | P lower (r2: these are the K1 / Z +10.4 / +10.6 % cells) |
| control | b8·1k, b8·4k, b8·16k | 16 | unchanged (the patch does not touch m ≤ 16) |
| control | b32·1k, b32·2k | 64 | unchanged |
| control | b1·1k, b1·4k, b1·16k | 2 | unchanged |

Not included, with reason: b8·K3 and b32·OFF also reach M = 32, but each needs its own config launches (×2–3 cost);
K3 also adds draft-decode steps. **Option for the user**: add K3 launches later only if K1 gives "lowers".

## 4. Launches and order (proposal)
6 measured launches, **0 P P 0 0 P** (ABBA then AB): 3 per arm, first-order position balanced except the last pair.
Before launch 1: one unmeasured warm launch (arm 0, one request; compile caches warm, identity check exercised).
The warm launch's numbers never enter the comparison.

## 5. Per-launch gates (stop the run on failure; no automatic re-rental)
1. Identity (gate-2 lesson), `arms.check_identity` right after the server answers `/v1/models`, before any cell:
   **every** process under the server (descendants by ppid plus its process group) that maps a file whose
   **basename equals** `_C_stable_libtorch.abi3.so` must map exactly one such file, not deleted, with sha256
   `f2b60575…`, and carry exactly the arm's selectors in `/proc/<pid>/environ` (`VLLM_TEST_SM90_FP8_PR66852=1` for P;
   none of `VLLM_TEST_SM90_FP8_M128_CAND`, `VLLM_TEST_SM90_FP8_PR66852`, `VLLM_BATCH_INVARIANT` for 0); at least one
   process must map it. *(Changed in §9 from "the EngineCore process": 7ecc names it through `setproctitle`, which is
   skipped silently when the package is missing, so the check does not depend on the title; it covers EngineCore and
   any other process that loads the kernels.)* Failure → server stopped, launch `stopped:IDENTITY_FAILED`, run STOPPED.
2. Server log at the same moment: `Selected CutlassFP8ScaledMMLinearKernel for …` present (r2 logs have exactly one
   such line, for `CompressedTensorsW8A8Fp8`); all `Selected <kernel> for <scheme>` lines recorded in
   `identity.json`. The drafter prints no such line in r2 (unquantized), so "the selector does not reach the drafter"
   rests on that absence, not on a positive record. MRv2 / `TRITON_ATTN` stay in the end-of-launch gate (3).
3. Existing r2 launch / early gates and per-cell validity (`analyze_costmap`: ≥ 40 steady intervals, FULL target
   replay ≥ 99 %, verify width = K, no preemption) unchanged.

## 6. Measures
- **Primary**: per (launch, cell) `step_period_ms` from `analyze_costmap` (non-profiled, steady decode window).
- Recorded beside it (not gated): target-forward `graph_tokens` per steady step (must be 32 / 16 / 64 / 2 as above;
  otherwise the cell is reported as "M not as planned" and excluded from the verdict), acceptance rate, wait vs host
  split (as in the b16 trace check), sentinel drift.
- Acceptance: gate 2 found P bitwise equal to arm 0 at M ≤ 32 in isolation, so equal acceptance is expected; an arm
  difference > 0.01 is reported (engine nondeterminism, #55265) and does not by itself void a cell — in the steady
  K1 window verify width and graph size do not depend on acceptance.

## 7. Decision rule (proposal; fixed before any run)
Per cell c: x_a,i = log step_period of arm a in its launch i (i = 1..3).
d(c) = mean_i x_0,i − mean_i x_P,i (positive = P faster).
s(c) = max(range_i x_0,i, range_i x_P,i) — **within-arm launch spread, per cell**. Gate 2 showed that one arm's
pass difference can underestimate between-arm differences at unchanged points (M 129 / 192 breach); here each arm has
three launches and the controls test the between-arm difference directly.

1. **Test**: P lowers the M = 32 step if, at **both** b16·1k and b16·4k: d > max(2·s, 3 %) **and** every P launch is
   below every arm-0 launch (3 vs 3 complete separation).
   In log units 3 % is −log(0.97) = 0.0305 (P at least 3 % below arm 0, as gate 2's "3 % of S_0"); 2 % is −log(0.98).
2. **Controls**: every control cell has |d| ≤ max(2·s, 2 %). One breach → verdict "not attributable" whatever (1) says.
3. Reported, not gated: saved ms at b16 = mean step_0 − mean step_P, against the isolated estimate 2.05 ms @ M 32
   (gate 2, Gemma-4 call counts) and against r2's K1 / Z jump at b16 (+2.3 / +2.5 ms, 1k / 4k).

4. Missing data (user, 2026-11-19; replaces the §9 draft of this rule). A cell is valid when each arm has three valid
   launches and its steady steps ran at the planned M. Only valid controls are evaluated, and the verdict always states
   `n/8` valid controls with each missing one's reason. Applied in this order:

   | situation | verdict |
   |---|---|
   | b16·1k or b16·4k without three valid launches per arm, or at an M other than 32 | effect not evaluable (`not_evaluable`) |
   | any valid control beyond its limit (rule 2) | not attributable to the config (`not_attributable`) |
   | every valid control within its limit, but n < 8 | observed change at M = 32 and the missing reasons reported; attribution not evaluable (`controls_incomplete`) |
   | 8 valid controls and both b16 cells valid | rules 1–2 as above (`lowers` / `not_detected`) |

| outcome | statement | next (decision (120)) |
|---|---|---|
| lowers, controls hold | "Under these conditions, the #66852 (16, 32] config lowered the server step at M = 32 by x %; controls unchanged." | user decides public record; K3 option |
| not detected, controls hold | "No step change beyond the registered threshold was detected." ⛔ not "no effect" | close the kernel branch, return to DepthChart |
| not attributable | "Arms differ at cells the config does not change; cause open." | no automatic re-run; user decides |
| controls incomplete (rule 4) | "Every evaluated control held, but not all 8 were evaluable; … attribution not evaluable." + observed change + n/8 | user decides |
| gate stop / test cell not evaluable | no verdict | user decides |

Power note: r2's K1 launch spread at b16 was 0.5–3.0 %; a full transfer of the isolated estimate is ≈ 8 % of a
≈ 25 ms step. An effect below ≈ 3–6 % can be missed by this rule; "not detected" is stated as such.

## 8. Cost (estimate; offer and credit re-read at launch)
Per launch ≈ 140 s start + ≈ 460 s cells (r2 K1 wall 463 s) ≈ 10 min; 6 + warm ≈ 65 min, + provision / model fetch
≈ 15 min ⇒ ≈ 80 min ≈ $3.9–4.5 at $2.9–3.4/h, × 1.2 ≈ $4.7–5.4. Vast credit 13.22, balance 0, 0 instances
(2026-11-19 09:02 KST).

## 9. Work before registration — done 2026-11-19 (CPU / local only), checks in `costmap_20261107/cpu_checks/server_cmp/`
| item | where | check |
|---|---|---|
| 1 runner arms | `run_costmap.py` (`ARMS`, `ARM_SO_SHA256`, `WARM_ARM`; dirs `<config>_<arm>_r<rep>`; `arm` in `launches.jsonl`) | unarmed path: existing `check_runner.py` 7/8 identical, 1 known timing-dependent reason |
| 1 analysis | `compare_arms.py` (§7 incl. rule 4 as revised by the user), `test_compare_arms.py` | 14 passed; end to end on a stub run (`controls_incomplete`, 7/8: the stub preempts b8·16k by design) |
| 2 identity | `arms.py` (`check_identity`, `arm_env`) | `--selftest` 13/13 on fake /proc; runner level `stub/check_arms_runner.py` |
| 3 box | `server_cmp/run_server_cmp_box.sh`: packet .so hash → provision → preflight (`--download`, as r2) → swap + hash → `so_load_check.py` (hash of the mapped file, one op, `check_identity` on its own process = real /proc parsed before any server) → runner | `stub/check_server_cmp_box.sh` 7/7 |
| 4 rehearsal | `stub/check_arms_runner.py` | normal, selector in runner env, wrong .so, child drops selector, no CUTLASS line, ARMS mismatch, unarmed |
| 5 packet | `server_cmp/make_packet_srv.sh` (committed files + .so by hash; `pilot/make_packet.sh` untouched), `server_cmp/start_box_srv.sh` | manifest re-checked after extraction (§10) |
| 6 lessons | `pilot/finish_pilot.sh`: one tar stream, per-file scp only as fallback; `finish.pid` kill only | `stub/check_scripts.sh` same outcomes, `collected (tar stream …)` |

Not done here: Linux /proc formats (first exercised by `so_load_check.py` on the box), vLLM's real process tree,
the models' download on the box (as in r2 / P1: same pins, same `preflight --download`; not separately rehearsed
off the box — the model files cannot be staged elsewhere for this host), GPU timing.
The engine tree is r2's (`eeb37ef757`), so r2's static import check (`cpu_checks/static_symbols_r2.json`) stands;
the new modules import only the standard library, numpy and the packet's own modules (`so_load_check.py` also torch and
`vllm._custom_ops`, as preflight does).

Order: §9 done → registration tag on the commit that adds the packet's manifest here → offer CHECK_ONLY → user's
explicit offer approval → rental.

## 10. Registration (2026-11-19)
- User 2026-11-19: §3 / §4 / §7 approved; rule 4 revised as in §7.4; P scope checked (`P_BRANCH_SCOPE_20261119.md`);
  "registration tag and offer check may proceed; rental only after the explicit offer approval".
- **The run uses the packet built from code commit `e0e36c7`** (`costmap_20261107/server_cmp/make_packet_srv.sh`):
  36 files, **manifest sha256 `5871f352af2876db9d398db4c4dc11cf582d2266db92603f24011f5dbcffbadb`**; tar kept untracked as
  `costmap_20261107/server_cmp/server_cmp_packet.tar` (its own sha differs per build; the box verifies the manifest).
  Re-checked after extraction: 36/36 hashes, manifest equal, imports and compile OK, `compare_arms.py` / `arms.py` /
  `run_costmap.py` byte-equal to the commit, engine patch sha `85f25f7e…` (r2's), .so `f2b60575…`.
- Between `e0e36c7` and the tag commit only documents and the cost-map preview changed (`kernel_probe_20261117/…`),
  none of them in the packet.
- Tag `server-cmp-66852-prereg-20261119` on the commit that adds this section. Anything changed after it is a dated
  note here, never a silent edit.
- Next: offer CHECK_ONLY (`create_and_arm_smoke.sh`, CEILING / FAST_MIN = user) → user's explicit approval of that
  offer → create → `start_box_srv.sh`.
