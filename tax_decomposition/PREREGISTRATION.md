# Graph-mode tax decomposition (V1 PIECEWISE vs V2 FULL) — pre-registration

- Track: DepthChart / DSD tax attribution (#49986) + #49652 third-party verification
- Status: PRE-REGISTRATION v2 — results not yet collected. §2 scope, §4 predictions, §6 decision rules are frozen; no retroactive edit after data.
- Date: 2026-08-05 (v2 supersedes v1: primary vehicle changed from proxy dense+EAGLE-3 to the original stack on the #49652 branch)
- Prior: #49986 (−31% dsd-K0 vs no_spec, Gemma-4-31B hybrid + MTP, ctx 400), #48494 (the assert we hit on V2), #49652 (fix, OPEN @ `fd355781`), PR #48944, RFC #48627

---

## 0. One line

Decompose the #49986 −31% DSD tax into its graph-mode (PIECEWISE→FULL cudagraph) and residual (spec bookkeeping + drafter) components, **on the original stack itself** — Gemma-4-31B hybrid + MTP — by holding model/draft/workload fixed and toggling only the runner: V1 (main, PIECEWISE) vs V2 (the #49652 branch, FULL). No proxy.

## 1. Why this is now possible without waiting for merge

Testing an open PR ≠ waiting for it to merge. The #49986 stack could not run V2 because autoregressive draft decode under dynamic SD produced invalid query lengths and hit `assert 0 < num_reqs <= num_tokens` (#48494). #49652 fixes exactly that (`use_dynamic_decode_query_len`, disables the derivation only for the draft-decode manager). Static verification (fresh main `ffee32460`):
- Hybrid is **not** a V2-unsupported feature — `_get_v2_model_runner_unsupported_features` (L2197-2235) lists CP / stock torch.compile / SP / PP-external / ngram / non-whitelisted spec method / P-Eagle; **hybrid is absent**. L669 only excludes hybrid from *default* V2 selection; `VLLM_USE_V2_MODEL_RUNNER=1` (L597-599) overrides.
- MTP is a V2-whitelisted spec method (L2222).
→ Gemma-4 hybrid + MTP + `VLLM_USE_V2_MODEL_RUNNER=1` on the #49652 branch should pass config validation and clear the #48494 assert at runtime.

## 2. SCOPE (frozen)

> This experiment IS the direct component attribution of the #49986 −31% (same model, same MTP draft, same workload). The only thing that changes between the compared arms is the runner (V1 PIECEWISE ↔ V2 FULL). **Provenance caveat (replaces v1's proxy caveat):** the V2 arm is built on the *unmerged* #49652 branch pinned at `fd355781`; results are labelled "V2 via #49652@fd355781" and are contingent on that branch, which may change. If the branch fails to open our stack on V2 (a second blocker beyond #48494), that failure is a reportable third-party finding for #49652, not a silent fallback.

The v1 proxy scope restriction ("NOT the −31% attribution") is lifted because the vehicle is no longer a proxy.

## 3. IN-SESSION GATES (frozen, ordered)

**Gate B (branch opens our stack on V2) — first, before any measurement.** Build #49652 @ `fd355781`, launch Gemma-4-31B hybrid + MTP with `VLLM_USE_V2_MODEL_RUNNER=1`. Confirm it reaches steady decode (no assert, no other blocker) and spec activation shows on `/metrics`.
- Pass → proceed.
- Fail (second blocker) → **capture the exact failure (file:line, config), that is the primary deliverable to #49652/CZT0.** Then fall back to the v1 proxy (dense + EAGLE-3, `VLLM_USE_V2_MODEL_RUNNER=0/1`) for a scoped graph-mode existence result with the v1 caveat re-applied.

**Gate T (tax existence) — before spending V2 arm time.** On V1, measure `no_spec` vs `spec` (K fixed) at the primary ctx. This is the #49986 −31% stack, so tax is expected; this gate is a sanity re-confirmation on the current build.
- ≥ ~10% slower → proceed to V2 decomposition.
- ≲ 5% (tax vanished on current main vs the #49986 measurement) → investigate the regression before decomposing; do not assume the old −31% still holds on today's main.

## 4. Cost model & pre-registered predictions (frozen)

Per-decode-step, spec on: `T_spec ≈ T_target_verify(runner) + T_draft + T_bookkeeping(runner)`. Only `runner` differs between the compared arms.

- **P1 (graph-mode component non-zero):** Tax(V2) < Tax(V1) by a margin exceeding noise; threshold ≥ 3%p of no_spec-normalized tax recovered.
- **P2 (direction):** the margin is a reduction of the spec-vs-no_spec gap, not a shift of the no_spec baseline. Report both arms' no_spec; if V1/V2 no_spec disagree beyond noise, the deltas are confounded → report runner-baseline-dependent, do not attribute.
- **P3 (ctx carry):** at long ctx the KV term dominates; the graph-mode share shrinks. Report share(short) vs share(long).
- **P4 (residual = the #48944-relevant number):** Tax(V2) is what remains after FULL cudagraph — spec bookkeeping + drafter. This is **what #48944's K=0 tier still pays after the V2 migration**, and the ceiling of the #49652/hybrid-V2 payoff. Name it explicitly.

## 5. Design

### 5.1 Vehicle
- **Primary:** original stack — Gemma-4-31B(-it-qat-FP8, as in #49986) target + its MTP draft. V1 arm on main; V2 arm on #49652 @ `fd355781` with `VLLM_USE_V2_MODEL_RUNNER=1`.
- **Fallback (only if Gate B fails):** dense (Llama-3-8B) + EAGLE-3 (`parallel_drafting=False`), toggled `VLLM_USE_V2_MODEL_RUNNER=0/1`. Scoped graph-mode-existence result with the v1 proxy caveat.
- **Not dspark** (force-V2, no V1 arm).

### 5.2 Arms
| arm | build / runner | spec |
|---|---|---|
| A | main, V1 (PIECEWISE) | no_spec |
| B | main, V1 | dsd (MTP, K per #49986) |
| C | #49652@fd355781, V2 (FULL) | no_spec |
| D | #49652@fd355781, V2 | dsd (MTP, same K) |

Tax(V1)=(B−A)/A, Tax(V2)=(D−C)/C. Graph-mode component = Tax(V1)−Tax(V2). Residual = Tax(V2).
(Ideally A/C use the same build with only the env toggle; if the #49652 branch is required to even launch our stock config, note that A/B use main and C/D use the branch, and confirm A(main-V1) ≈ a branch-V1 spot check to rule out branch-side baseline drift.)

### 5.3 Grid
- ctx ∈ {short ~400 (the #49986 point), long ~4000}.
- Fixed K (the #49986 dsd-K0 and one K>0 point; runner axis is primary, not a K sweep).
- Metric: **TPOT p50/p99** primary; tok/s secondary. 3 warmup discarded + 3 measured per cell.

### 5.4 Methodology (inherited discipline)
autotune cache wipe per server / cold-start burn / PYTHONHASHSEED=0 / position-balanced prompts / container stop-start between a compared pair invalidates it / spec activation proven via `/metrics` draft counter before measuring / preemption counter watched / decision rules pre-registered here / raw JSON + build SHAs preserved and pushed to depthchart.

**Dup-search / prior-art discipline (methodology note, 2026-08-05):** before any public artifact claiming novelty or absence, run an authenticated search **with a positive control** (a query known to return hits) to prove the search functions, then targeted token queries. An empty result is "absent" only after the positive control fires. (This caught a false-negative dup clearance for the routed-experts bug this session.)

## 6. Decision rules (frozen)
| Gate B | Gate T | P1 | Conclusion | Landing |
|---|---|---|---|---|
| pass | tax exists | Tax(V2)<Tax(V1) by ≥3%p | **−31% is X%p graph-mode + Y%p residual** (direct attribution) | #49986 update with the split; #48944 note: K=0 tier still pays Y%p after V2 |
| pass | tax exists | V1≈V2 (<3%p) | tax is bookkeeping/drafter, not graph-mode | #49986 update; downward-revises the hybrid-V2 payoff estimate |
| fail | — | — | second V2 blocker on our stack | **primary deliverable: #49652 third-party verification report** (exact failure) + fallback proxy run |

## 7. Limitations
- V2 arm on an unmerged branch (`fd355781`); pin SHA, re-verify at rental, results contingent.
- FULL vs PIECEWISE capture coverage may differ by attention backend; record the backend.
- Single stack; the residual P4 is this stack's, though it is the one #48944 cares about.

## 8. Budget
1 rental, 3–4h, ~$7–8. Gate B (build + launch) first, then Gate T, then C/D. HF_TOKEN pre-set (`hf auth token`, account jamesyun).

## 9. Roadmap position
This experiment sits directly on the track's performance-contribution target — **hybrid V2 enablement**. Hybrid (Gemma-4) can't use V2 → no FULL cudagraph → pays the tax; we are the only third party that has run hybrid+MTP+DSD on V2 and knows where it dies (#48494, measured). #49652 is the first domino; verifying its branch on our stack (Gate B) is itself that contribution, and finding/fixing the next blocker after it merges is a *merged* performance contribution that recovers our own −31%. **#49652 = start signal; testing its open branch is the start, not the wait.**
