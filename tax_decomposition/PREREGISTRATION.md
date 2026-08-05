# Graph-mode tax decomposition (V1 PIECEWISE vs V2 FULL) — pre-registration

- Track: DepthChart / DSD tax attribution (#49986) + #49652 third-party verification
- Status: PRE-REGISTRATION v3 — results not yet collected. §2 scope, §4 predictions, §6 decision rules are frozen; no retroactive edit after data.
- Date: 2026-08-05 (v3 supersedes v2: all 4 arms on a single build (#49652 branch) toggling only the env; added a build-drift spot check; added the Tax(V2)>Tax(V1) and Gate-T-null branches to the frozen matrix; separated K=0 tax from K>0 net-effect)
- Prior: #49986 (−31% dsd-K0 vs no_spec, Gemma-4-31B hybrid + MTP, ctx 400), #48494 (the assert we hit on V2), #49652 (fix, OPEN @ `fd355781`), PR #48944, RFC #48627

---

## 0. One line

Decompose the #49986 −31% DSD tax into graph-mode (PIECEWISE→FULL cudagraph) and residual (spec bookkeeping + drafter) components, **on the original stack** (Gemma-4-31B hybrid + MTP), by holding model/draft/workload/**build** fixed and toggling only the runner via `VLLM_USE_V2_MODEL_RUNNER=0/1`. No proxy.

## 1. Why this is possible without waiting for merge

Testing an open PR ≠ waiting for it to merge. The #49986 stack could not run V2 because autoregressive draft decode under dynamic SD produced invalid query lengths and hit `assert 0 < num_reqs <= num_tokens` (#48494). #49652 fixes exactly that (`use_dynamic_decode_query_len`, disables the derivation only for the draft-decode manager). Static verification (fresh main `ffee32460`):
- Hybrid is **not** a V2-unsupported feature — `_get_v2_model_runner_unsupported_features` (L2197-2235) lists CP / stock torch.compile / SP / PP-external / ngram / non-whitelisted spec method / P-Eagle; **hybrid is absent**. L669 only excludes hybrid from *default* V2 selection; `VLLM_USE_V2_MODEL_RUNNER=1` (L597-599) overrides.
- MTP is a V2-whitelisted spec method (L2222).
→ Gemma-4 hybrid + MTP + `VLLM_USE_V2_MODEL_RUNNER=1` on the #49652 branch should pass config validation and clear the #48494 assert at runtime.

## 2. SCOPE (frozen)

> This experiment IS the direct component attribution of the #49986 −31% (same model, same MTP draft, same workload, **same build**). The only thing that changes between the four arms is the runner (V1 PIECEWISE ↔ V2 FULL), set by the env var. **Provenance caveat:** the build is the *unmerged* #49652 branch pinned at `fd355781`; results are labelled "on #49652@fd355781" and are contingent on that branch, which may change. Because #49652 alters the DSD query-length derivation, a build-drift spot check (§5.2) confirms the branch's V1 path matches main before any attribution to #49986. If the branch fails to open our stack on V2 (a second blocker beyond #48494), that failure is a reportable third-party finding for #49652, not a silent fallback.

## 3. IN-SESSION GATES (frozen, ordered)

**Gate B (branch opens our stack on V2) — first.** Build #49652 @ `fd355781`, launch Gemma-4-31B hybrid + MTP with `VLLM_USE_V2_MODEL_RUNNER=1`. Confirm steady decode (no assert, no other blocker) + spec activation on `/metrics`.
- Pass → proceed.
- Fail (second blocker) → **capture exact failure (file:line, config) = primary deliverable to #49652/CZT0.** Then fall back to a proxy (dense + EAGLE-3, `parallel_drafting=False`, env toggle) for a scoped graph-mode-existence result, re-applying a proxy caveat.

**Gate T (tax reproduces on today's build) — before spending V2 arm time.** On branch-V1, measure `no_spec` vs `dsd-K0` at the #49986 ctx point.
- ≥ ~10% slower → tax reproduces → proceed to V2 decomposition.
- ≲ 5% → **our published −31% no longer reproduces on current main.** This is not "investigate later" — it is a **#49986 correction event** (see §6). Report it; do not proceed to attribute a tax that no longer exists.

## 4. Cost model & pre-registered predictions (frozen)

Per-decode-step, spec on: `T_spec ≈ T_target_verify(runner) + T_draft + T_bookkeeping(runner)`. Only `runner` differs between the four arms.

**Tax is defined on the K=0 arm only.** dsd-K0 is pure overhead (no drafting, no acceptance gain) → it isolates the tax. A K>0 arm mixes overhead with acceptance benefit and is NOT a tax measurement; it is reported separately as "net effect" (§5.3), never in the Tax column.

- **P1 (graph-mode component non-zero):** Tax(V2) < Tax(V1) by a margin exceeding noise; threshold ≥ 3%p of no_spec-normalized tax recovered.
- **P2 (direction / baseline integrity):** the margin is a reduction of the spec-vs-no_spec gap, not a shift of no_spec. Report both arms' no_spec; if they disagree beyond noise the deltas are confounded → report runner-baseline-dependent, do not attribute.
- **P3 (ctx carry):** at long ctx the KV term dominates; graph-mode share shrinks. Report share(short) vs share(long).
- **P4 (residual = the #48944 number):** Tax(V2) is what remains after FULL cudagraph — spec bookkeeping + drafter. This is **what #48944's K=0 tier still pays after the V2 migration**, and the ceiling of the hybrid-V2 payoff. Name it explicitly.

## 5. Design

### 5.1 Vehicle
- **Primary:** Gemma-4-31B(-it-qat-FP8, as in #49986) target + its MTP draft, **all arms built from #49652 @ `fd355781`**.
- **Fallback (only if Gate B fails):** dense Llama-3-8B + EAGLE-3 (`parallel_drafting=False`), env toggle. Scoped, proxy caveat.
- **Not dspark** (force-V2, no V1 arm).

### 5.2 Arms — single build, env is the only lever
| arm | build | `VLLM_USE_V2_MODEL_RUNNER` | runner | spec |
|---|---|---|---|---|
| A | #49652@fd355781 | 0 | V1 PIECEWISE | no_spec |
| B | #49652@fd355781 | 0 | V1 | dsd-K0 (MTP) |
| C | #49652@fd355781 | 1 | V2 FULL | no_spec |
| D | #49652@fd355781 | 1 | V2 | dsd-K0 (MTP) |

Tax(V1)=(B−A)/A, Tax(V2)=(D−C)/C. **Graph-mode component = Tax(V1)−Tax(V2). Residual = Tax(V2).** Because A/B/C/D share one build, `runner` is the single variable — no build axis confound.

**Spot check S (build-drift + #49986 linkage, ~30 min, before the grid):** one cell of `dsd-K0` on **main-V1** at the #49986 ctx point. Purpose: (a) confirm branch-V1 (arm B) ≈ main-V1 — since #49652 touches DSD query-len derivation, the branch's V1 path could differ; (b) tie the branch tax to the published #49986 −31% **transitively** — S alone does not confirm the −31%; the chain does: Gate T measures the branch tax, S confirms branch-V1 ≈ main-V1, therefore main tax ≈ branch tax ≈ published −31%. If S(main-V1) ≠ B(branch-V1) beyond noise, the branch altered V1 behavior → report and re-scope.

### 5.3 Grid
- ctx ∈ {short ~400 (the #49986 point), long ~4000}.
- **Concurrency (registered pre-data, 2026-08-05 addendum, KV-margin):** ctx400 @ c=256; ctx4000 @ **c=192**. KV pool at mml 8192 / util 0.90 / 94GB = 55,215 tokens (log-measured from `pr49986_runbook/raw/logs/vllm_v2.log`); ctx4000 @ c=256 working set ≈ 4000 + 256×196 ≈ 54,176 leaves ~1k margin → preemption. c=192 (≥129, K=0 tier preserved) → ≈41,632, comfortable. Applied **uniformly to all 4 arms at each ctx** (else within-ctx tax invalid); cross-ctx P3 compares shares not absolutes, so the c difference is fine.
- **Tax decomposition (canonical): K=0 only** — arms A–D above.
- **Net-effect (separate table, not "tax"):** one K>0 point (dsd-K vs no_spec) on V1 and V2, reported as end-to-end net effect (overhead + acceptance). Explicitly labelled distinct from the tax.
- Metric: **TPOT p50/p99** primary; tok/s secondary. 3 warmup discarded + 3 measured per cell.

### 5.4 Methodology (inherited discipline)
autotune cache wipe per server / cold-start burn / PYTHONHASHSEED=0 / position-balanced prompts / container stop-start between a compared pair invalidates it / spec activation proven via `/metrics` draft counter before measuring / preemption counter watched / decision rules pre-registered here / raw JSON + build SHA (`fd355781`) preserved and pushed to depthchart.

**Dup-search / prior-art discipline (methodology note, 2026-08-05):** before any public artifact claiming novelty or absence, run an authenticated search **with a positive control** (a query known to return hits) to prove the search functions, then targeted token queries. An empty result is "absent" only after the positive control fires. (This caught a false-negative dup clearance for the routed-experts bug this session.)

## 6. Decision rules (frozen — all branches fixed before data)
| Gate B | Gate T | outcome | Conclusion | Landing |
|---|---|---|---|---|
| pass | reproduces | Tax(V2) < Tax(V1) by ≥3%p | −31% = X%p graph-mode + Y%p residual (direct attribution) | #49986 update with split; #48944 note: K=0 tier still pays Y%p after V2 |
| pass | reproduces | \|Tax(V1)−Tax(V2)\| < 3%p | tax is bookkeeping/drafter, not graph-mode | #49986 update; downward-revises hybrid-V2 payoff estimate |
| pass | reproduces | **Tax(V2) > Tax(V1) by ≥3%p** | **V2/FULL is WORSE than V1/PIECEWISE on this hybrid stack** (FULL capture cost on hybrid, or V2 bookkeeping regression) | #49986 + **MRV2 team caution: V2 migration is not a win for hybrid+MTP; here is where** — high-value negative result |
| pass | **null (<5%)** | — | **published −31% does not reproduce on current main** | **#49986 correction: tax shrank between the original measurement and today; cause investigation is a separate item.** Do not attribute a vanished tax |
| fail | — | — | second V2 blocker on our stack | **primary: #49652 third-party verification report** (exact failure) + fallback proxy |

- Baseline guard: if branch V1/V2 no_spec (A vs C) disagree beyond noise, deltas are confounded → runner-baseline-dependent, do not attribute.

## 7. Limitations
- V2 arm on an unmerged branch (`fd355781`); pin SHA, re-verify at rental, results contingent.
- FULL vs PIECEWISE capture coverage may differ by attention backend; record the backend.
- Single stack; residual P4 is this stack's, though it is the one #48944 cares about.

## 8. Budget
1 rental, 3–4h, ~$7–8. Gate B (build + launch) → Gate T → Spot check S → C/D grid → net-effect table. HF_TOKEN pre-set (`hf auth token`, account jamesyun).

## 9. Roadmap position
This experiment sits on the track's performance-contribution target — **hybrid V2 enablement**. Hybrid (Gemma-4) can't use V2 → no FULL cudagraph → pays the tax; we are the only third party that has run hybrid+MTP+DSD on V2 and knows where it dies (#48494, measured). #49652 is the first domino; verifying its branch on our stack (Gate B) is itself that contribution, and finding/fixing the next blocker after it merges is a *merged* performance contribution that recovers our own −31%. **#49652 = start signal; testing its open branch is the start, not the wait.**
