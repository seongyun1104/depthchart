# HANDOFF — speculative-decoding context-axis track

Last updated: 2026-08-10.

Coordination pointer for the dynamic-SD / context-axis work. Every claim below is
anchored to a committed SHA or a live GitHub artifact. Numbers live in the
committed `RESULTS.md` / raw files, not in prose here.

## Thesis (canonical vessel: RFC vllm-project/vllm#48627)

Speculative-decoding K / verification budget should treat context length as an
explicit policy axis, not only an implicit cost-model term. Sharpened form: no
batch-only allocation `K(batch)` can be simultaneously optimal at short and long
context — for a fixed batch, `K*(b, 4k) ≠ K*(b, 32k)`. This is an expressiveness
argument, not a "retune the table" gap.

## Landed artifacts

### PR vllm-project/vllm#48944 — (batch, ctx) dynamic-SD schedule
- Branch `feat/dsd-2d-ctx-schedule` on the fork, head `dbd51d4b6`, OPEN / MERGEABLE (2026-08-10).
- Config field renamed `num_speculative_tokens_per_batch_size` -> `speculative_token_schedule`;
  the old name is kept as a deprecated pydantic validation alias (emits `DeprecationWarning`).
- 5-item entries `(bs_lo, bs_hi, ctx_lo, ctx_hi, K)` add the ctx axis to the legacy
  3-item `(bs_lo, bs_hi, K)`; `ctx_agg` in {median, mean, max}, default `mean`
  (per the attention microbenchmark).
- Latest status comment: #48944 issuecomment-5232632536. Awaiting reviewer (benchislett)
  decision on the name, plus CI. pytest not run locally (no macOS wheel) — CI validates.

### Measurement results (this repo)
- **P1 campaign** — position-balanced 2-trial `C'/A'` = 1.29x / 1.30x / 1.36x at
  ctx 900 / 1900 / 4000; C' crosses no-spec near ctx ~2k. Pre-registered SUCCESS.
  Posted at #48944 (2026-07-27).
- **Baseline-tax decomposition** (`tax_decomposition/RESULTS.md`) — H100 NVL,
  Gemma-4-31B hybrid+MTP on the #49652 branch: `Tax(V1) = +7.29%`,
  `Tax(V2) = +16.64%`, baseline guard `(C-A)/A = -0.39%`. K=0 arm only; K>0 unmeasured.
  Posted at #49986.
- **ctx uplift spine** (`ctx_uplift/RESULTS.md`, commits `bab2417` + `08a9b59`) —
  fixed-batch b=11, ctx {4k, 32k}, K {0,1,2,3,5,7}: `K*(4k)=3`, `K*(32k)=5`
  (optimal K rises with context = thesis direction) but max forcing loss `+4.51%`,
  far below the pre-registered 15% bar -> **WEAK**. The 32k argmax sits inside
  ~10% per-seed noise; the 4k anchor `tax(K0 vs no_spec)@4k = +22.19%` is out of
  band (b=11 vs #49986 b=192, post-hoc, not comparable).

## Live issue/PR status (re-verify before acting)
- **#48627** RFC — thesis vessel. ekagra "Motivation makes sense / follows MagicDec".
  External comment from elmehdi-eljair (MemForge; self-promotional) echoing the
  operating-surface framing, 2026-08-09.
- **#49986** — three-stack tax thread: our H100 dense-hybrid + Suppressor72
  (SM120 / MoE / TP=2, 2026-08-08) + elmehdi-eljair `T_DSD` term list (2026-08-09).
  Our reply issuecomment-5226922413. Suppressor72 collaboration reply pending.
- **#49652** — CZT0 V2+DSD enablement, OPEN / REVIEW_REQUIRED (maintainers pinged,
  not merged). Our Gate B verification: issuecomment-5189630935. Merge = trigger
  for the V2+DSD rental.
- **#51466** — OPEN, K-lookup thrashing fix; touches the same `scheduler.py` as #48944.
- **#47808** — DSpark (Lucas). Our role here is hardware verifier / data supplier,
  not policy designer.

## Open experiments (with pass criteria)
1. **ctx sweep 32k precision** — the 32k argmax is inside first-seed spec-warmup
   noise. Fix = more warmup rounds (not more seeds). Pass = `K*(32k)` separated
   from `K*(4k)` at >2σ with forcing loss >= 15%.
2. **Phase B — K>0 tax** — decompose the tax for K>0 (V2+DSD), separating the
   MRV2-runner cost from the #49652 patch cost via `use_dynamic_decode_query_len`
   True/False on the #49652 branch. Blocked on #49652 (V2+DSD otherwise hits the
   #48494 assert).
3. **#51466 verification** — verify the K-lookup thrashing fix on a rig (the H3
   factor from #49986).

Rental prerequisite: Vast credit is $0 -> top up first. Runbooks:
`ctx_uplift/RUNBOOK.md`, `tax_decomposition/RUNBOOK.md`.

## Watch triggers
- #49652 merged -> V2 3-check + Phase B rental.
- benchislett responds on #48944 naming -> apply / adjust the rename.
- Suppressor72 responds on #49986 -> P2' cross-stack collaboration.

## Working norms (index)
- Report only from live remote artifacts; status predicates ("posted / merged /
  landed") carry a same-turn URL or SHA, and attach the raw command output.
- Outward vllm-project mutations (posting / deletion) are executed by the human,
  with the session preparing the command and verifying live afterward. Pushes to
  our own fork are session-run and checked with `git ls-remote`.
- Verify local git identity (`seongyun.kim` / `197560810+seongyun1104`) before
  every public commit.
- API renames / breaking changes: public declaration + wait before applying;
  silence is not consent.
- Pre-register success criteria before measuring; report WEAK as WEAK.
