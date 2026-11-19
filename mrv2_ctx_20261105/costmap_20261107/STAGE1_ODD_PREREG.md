# Stage 1 (S1-odd) pre-registration

Chosen option: **S1-odd** of `STAGE1_DESIGN_v2_20261109.md` (user, 2026-11-09). Fixed by the
registration commit and tag named at the end; anything changed after the tag is a dated note
here, never a silent edit. No rental is implied: top-up and CEILING are the user's.

## 1. Question and scope

DepthChart asks whether a **BS×CTX K controller** beats a **good fixed K** and a **batch-only
controller**, including its own cost. Stage 1 measures the cost side only: step period and time per
delivered token for OFF, Z, K1, K3, K5 at shapes where batch-only and BS×CTX would choose
differently, and whether a difference exceeds launch-to-launch spread.

Not claimed by stage 1: the marginal cost of one extra K between adjacent K (K2, K4 not run), an
optimal K among all candidates, any speedup, any policy result. Pilot 2 is not re-judged.

## 2. Cells (deduplicated) and what one launch runs

One launch = one server start with one configuration, then 13 measurements in the order of
`plan(20261109, "stage1")`:

| # | id | BS × ctx | role |
|---|---|---|---|
| 0 | warmup_b8_c4096_9000 | 8 × 4k | warm-up, discarded from comparisons |
| 1 | grid_b1_c16384 | 1 × 16k | shape |
| 2 | grid_b32_c1024 | 32 × 1k | shape (high BS) |
| 3 | grid_b16_c4096 | 16 × 4k | shape |
| 4 | grid_b1_c4096 | 1 × 4k | shape |
| 5 | grid_b8_c1024 | 8 × 1k | shape |
| 6 | sentinel_b8_c4096_9001 | 8 × 4k | drift repeat 1 |
| 7 | grid_b8_c4096 | 8 × 4k | shape |
| 8 | grid_b32_c2048 | 32 × 2k | shape (high BS) |
| 9 | grid_b8_c16384 | 8 × 16k | shape |
| 10 | grid_b1_c1024 | 1 × 1k | shape |
| 11 | grid_b16_c1024 | 16 × 1k | shape |
| 12 | sentinel_b8_c4096_9002 | 8 × 4k | drift repeat 2 |

Ten distinct shapes, each measured **once per launch**. b8 × 4k appears four times per launch:
once as the shape (#7, compared) and three times as repeats (#0 discarded, #6/#12 used only for
the in-launch drift `log(#12 / #6)`), never pooled with #7. Bench seeds: shapes 20000-20009,
repeats 9000-9002, identical across configs and launches.

Configurations and launches (`CONFIGS` in `stage1/run_stage1_box.sh`):

| order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| config | K3 | K1 | K5 | OFF | Z | Z | OFF | K5 | K1 | K3 |
| rep | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |

A = `random.Random(20261109).shuffle(["OFF","Z","K1","K3","K5"])`, B = A reversed: every config runs
exactly **twice** (rep 0 in A, rep 1 in B); the pair is the config's A/A. 10 launches × 13 = 130
measurements, 100 of them shapes.

## 3. OFF and Z

| | server flags | drafter loaded | verify width | graph sets |
|---|---|---|---|---|
| OFF | no `--speculative-config` | no | 0 | target only |
| Z | `{"model": draft, "num_speculative_tokens": 1, "num_speculative_tokens_per_batch_size": [[1, 64, 0]]}` | yes | 0 at every BS 1-64 (dynamic table) | target + drafter, K_max = 1 captured |
| K1/K3/K5 | `{"model": draft, "num_speculative_tokens": K}` | yes | K | target q = K+1 + drafter |

Z − OFF is the cost of having the drafter present at K = 0 (memory, graph set, any per-step
drafter work); steady selection requires width 0 on every row for both (`verify_ok`).

## 4. Pilot 43/45 versus "4 excluded": two different units

Pilot 2 measured 50 cells = 47 unique shapes (36 grid + 11 dense) + 3 repeats. The launch gate
counts **unique shapes that were feasible**: denominator 45 = 47 − 2 preempted (b64 × 1k, b64 × 2k,
outside the denominator); valid 43; so the 2 shapes invalid **inside** the denominator are b64 × 4k
and b32 × 8k (insufficient steady steps). "4 excluded" counts **shapes not usable** = 2 outside the
denominator + 2 invalid inside it. 43/45 is a gate statistic of one K3 launch, not a pass of those 4
shapes; all 4 stay outside stage 1 (§2 of the v2 design) and are reported as infeasible at K3 /
output 640.

## 5. Reported values and denominators

Per measurement: as in the pilot (steady step period, CI, wait/outside-wait, graph paths, KV,
preemptions, acceptance, validity with reasons). Denominators, both reported, named separately:

- **primary — delivered**: `delivered_ms_per_token` = bench median TPOT (client side, delivered
  tokens, per request, whole decode incl. ramp) ÷ BS.
- **secondary — server**: `ms_per_output_token` = steady step period ÷ (BS × (1 + K × acceptance
  from server counters over the whole cell)); labelled approximate. Pilot 2 showed the counters
  2-3 tokens per request above delivered tokens.

Neither denominator is applied to pilot 2's verdicts (re-analysis of pilot 2 with this analyzer
changes no validity; only these fields are added).

## 6. Comparison and disagreement between launches (`compare_stage1`)

- Compared set = shapes valid for **every** config with two launches, in **both** launches; any
  other shape is listed under `not_compared` with its per-launch reasons.
- For configs a, b at a shape: d = log(mean b / mean a) over the two launches; spread = the larger
  of the two configs' |log(rep1 / rep0)|.
- **exceeds_aa**: |d| > spread and b − a has the same sign in rep 0 and rep 1.
- **inconclusive**: the sign differs between rep 0 and rep 1 (launches disagree) — reported as such,
  never averaged into a direction.
- **within_aa**: same sign but |d| ≤ spread.
- Applied separately to step period, delivered and server denominators; the step-period and
  delivered results are reported side by side, never merged.
- A config with only one complete launch (stop, deadline) is not compared at all.
- Statements stay per shape and per pair; a pattern across BS or ctx is described only as the set
  of shapes where it holds.

## 7. Gates and stops (unchanged from the pilot unless stated)

- Early gate on every launch (first 3 measured cells); failure stops the run; no automatic re-rental.
- Launch gate per launch, reported per cell with the failing condition (valid share ≥ 0.90 over
  feasible shapes is a gate, not a summary of the cells).
- A preempted shape skips larger BS at the same ctx in that launch (runner rule); recorded as a
  feasibility result for that config.
- Runner deadline: no launch starts unless its estimate × 1.2 fits; first estimate 900 s, then the
  longest observed launch.

## 8. Cost (estimate, not a cap)

≈ 136 min ≈ **$7.7 at $3.414/h, × 1.2 ≈ $9.3** (pilot walls; other K re-estimated after the first
launch). Offer, price and credit are re-queried before any rental; top-up and CEILING are the user's.

## 9. Registration

Code: `analyze_costmap.py` (`plan(seed, "stage1")`, `compare_stage1`, `delivered_ms_per_token`),
`run_costmap.py` (`PLAN`, `LAUNCH_EST_S`), `stage1/run_stage1_box.sh`, `pilot/make_packet.sh`.
CPU checks: `cpu_checks/pytest_analyzer.txt` (17), `cpu_checks/stage1_checks.json`,
`cpu_checks/runner_checks.json` (8 pilot scenarios re-run). Tag and manifest: see the registration
commit.

Registered 2026-11-09: code commit `55e3e7a`; packet built by `pilot/make_packet.sh` from that
commit: 22 files, **manifest sha256 `36defd0085955e2a61ccd8609d51d58c269616265dc8e74303aff34fe328d65e`**
(re-checked after extraction: 22/22 OK). Tag `costmap-stage1-prereg-20261109` on the commit that
adds this line. The tar itself is untracked (its sha differs per build); the box verifies the manifest.
