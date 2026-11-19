# Stage 1 (S1-odd) re-run r2 — pre-registration

Drafted 2026-11-11 (`STAGE1_R2_PREREG_DRAFT_20261111.md`, `333e5d9`), checked R1–R3 and registered by the tag named
in §8. Anything changed after the tag is a dated note here, never a silent edit. No rental is implied: offer,
CEILING and FAST_MIN are the user's. The registered tag `costmap-stage1-prereg-20261109` (`12e1e743`, manifest
`36defd00…d65e`), its packet tar and attempt 2 (`19b4bbc`, four launches K3/K1/K5/OFF r0 on tree 6003621c, stopped
at Z_r0) are unchanged.

## 1. What stays as registered
Question, cells, launch order, repeats, denominators, `compare_stage1`, early/launch gates, deadline rule:
`STAGE1_ODD_PREREG.md` §1–§7 unchanged. 10 launches K3,K1,K5,OFF,Z,Z,OFF,K5,K1,K3 (A then reversed).

**Attempt 2 is not merged.** Its four launches are a separate partial record on a different engine tree; they are
never pooled with, substituted for, or compared as A/A partners of the new launches.

## 2. Engine source (changed)
| | attempt 2 | r2 |
|---|---|---|
| vLLM commit (local, not pushed) | `8e5f0678a1` | `dc/bs-ctx-dsd-zfix` **`bd09764c45`** |
| patch file | `bs_ctx_8e5f0678a1.patch` | `smoke/box/bs_ctx_bd09764c45.patch`, sha256 `85f25f7e…6278d7` |
| tree after `git apply` on 7ecc83a9 | `6003621c…` | **`eeb37ef757…`** (= commit tree; temporary index) |
| base wheel | 7ecc83a9 dev784 | unchanged |

`vllm/` of `bd09764c45` is byte-identical to `0a3a2e9703` / `b0a6fff660`; `bd09764c45` adds only test updates.
Every config runs on the new tree, so no attempt 2 number carries over.

## 3. Evidence (and what it does not show)
- r3 CPU (11-10): Z prefill-capture TypeError reproduced on up/pre, absent on upfix/fix. Suite comparison N/A
  (pre/upfix timed out) — no "no additional regression" claim (`zfix_cpu_20261110_r3/analysis/`).
- R1 CPU (11-11, `zfix_cpu_20261111_r4/RESULT_R1_20261111.md`): six trees; the speculator manager tests fail only on
  the fix-without-test-update trees and pass on `bd09764c45`; capture tests as expected; trace hook 6/6 incl. the
  role test (query-length-1 decode vs prefill told apart by `draft_decode`).
- R3 static (`cpu_checks/static_symbols_r2.json`): every packet import resolves in tree `eeb37ef757`; 44 hook / gate
  contract checks pass (scheduler and output fields, graph manager methods and leading parameters, descriptor
  fields, engine wait methods, metric names, routes, request/usage fields, cache env vars). Negative control on the
  pre-fix tree `c80402d8` fails exactly the two `draft_decode` checks.
- **None of this is a real Z server start or a GPU capture** — that is the gate below.

## 4. Z start gate (before launch 1; not a measured launch) — `stage1/zgate.py`
Run once on the box after preflight, through `run_costmap.start_server` (same argv, trace hook, readiness check as
a measured launch). All compile/JIT caches redirected under `/root/zgate` (`HOME`, `XDG_CACHE_HOME`,
`VLLM_CACHE_ROOT`, `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`).
Pass requires, in order:
1. **start**: ready (`/v1/models`) within 900 s (attempt 2 cold start 120–215 s); `/health` 200.
2. **log**: no `Traceback` / `TypeError` in server.log.
3. **capture**: trace `capture` rows with descriptors for roles `target` and `drafter_prefill` (the attempt 2
   failure point). `drafter_decode` is recorded, not required: K_max = 1 has no draft decode capture, as in
   attempt 2's K1_r0 trace (target + drafter_prefill only). *(Correction of the draft, which required it.)*
4. **request**: one completion, fixed prompt, 16 tokens, temperature 0, `ignore_eos` → HTTP 200,
   `completion_tokens` 16; `vllm:request_success` +1 (this server's counters); `vllm:spec_decode_num_draft_tokens`
   +0; ≥ 1 new scheduler row and every one selected K = 0.
5. **stop**: process group gone, port closed, GPU memory within 512 MiB of the pre-gate reading.
6. **caches**: default cache trees (`~/.cache`, `~/.triton`, `~/.nv`, `/tmp/torchinductor_*`) unchanged
   file-by-file (path, size, mtime), then `/root/zgate` removed — so launch 1 starts from the cache state the
   registration assumed (no server before it, as in attempt 2).
- Any failure: `OUT/zgate/zgate.json` with the reason, `STOPPED "ZGATE rc=3 <reason>"`, the box script exits before
  the runner; finish script collects and destroys. No retry, no measured launch, no partial plan.
- Gate output never enters stage 1 comparisons (`launches.jsonl` does not list it).
- Stub checks (`cpu_checks/zgate_checks.json`): pass; server not ready (TypeError); no drafter_prefill capture; draft
  tokens at K=0; scheduler K=1 with zero drafts; default-cache leak — each stops at its own step with the expected
  reason, no process / port / gate dir left; box level: runner runs only after a pass, never after a fail.

## 5. After the gate
All 10 launches as registered on the new tree. Comparisons only among these 10 launches.

## 6. Cost (estimate, re-queried at launch)
≈ 10 × 8 + 17 + ≈ 5 (gate, cold start + capture + request) ≈ 102 min; at ≈ $3.4/h ≈ $5.8, × 1.2 ≈ $7.0.
FAST_MIN = registered value + 5 min. Offer, CEILING and FAST_MIN = user.

## 7. Checks behind this registration
R1 Linux CPU (Hyperstack VM 1154474, `539819b`) · R2 gate + stub checks (`zgate_checks.json`; existing
`stage1_checks` / `runner_checks` re-run unchanged, `server_exit`'s `missing_metrics_samples` is timing-dependent and
matched on the registered arguments with both stubs) · R3 static (`static_symbols_r2.json`) · R4 packet (§8).

## 8. Registration
Registered 2026-11-11: code commit `bea14ae`; packet built by `pilot/make_packet.sh` from that commit: 24 files
(the registered 22 + `costmap/stage1/zgate.py` + `smoke/box/bs_ctx_bd09764c45.patch`), **manifest sha256
`dd8f8c6964a2cd866921100b3011b03e8a603d720bad54a5ca4c12b9242c3d4a`**. Re-checked after extraction: 24/24 hashes OK,
every file equal to its committed blob, the packet's patch applied to 7ecc83a9 gives tree `eeb37ef757`, provision /
preflight / box script name that patch, sha and tree; base wheel URL 200 with the pinned 317,037,303 bytes.
Tag `costmap-stage1-r2-prereg-20261111` on the commit that adds this section. The tar is untracked (its sha differs
per build; kept as `pilot/costmap_packet_stage1r2_bea14ae.tar`); the box verifies the manifest.
