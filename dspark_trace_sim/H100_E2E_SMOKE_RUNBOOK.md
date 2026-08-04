# DSpark H100 E2E Smoke Runbook (진입점 2 = A)

First-verifier smoke: does #47808 adaptive verification actually run on H100
(SM90) with a dense Gemma-4-12B DSpark target? This is a **boot + activation +
correctness** smoke, not a throughput measurement.

> **Why this exists.** #6 source reading established that "SM100-only" for DSpark
> is a *benchmark scope*, not a code gate: `adaptive_verification.py` is
> backend-agnostic, and a dense Gemma target routes through FlashAttention (FA3),
> which admits SM90. That is a **static-source** claim. This smoke converts it to
> measurement. Only after this passes do we post the enablement comment on #47808
> ("your branch runs on H100 — here are the numbers and the one flag"); a
> static-only "it works on SM90" claim is measurement-free and off-identity.

## 0. Pin the branch (blocking, rental-day)

#47808 lives on `neuralmagic/vllm @ codex/dspark-capacity-realloc` and is
**rebased by mergify** — the head SHA moves. At rental start:

- Record the current head SHA (`git ls-remote` the branch). The last SHA we read
  from was `e399e1c7`; expect it to have moved.
- **Re-verify the two capability coordinates below still read as expected on that
  head** (they are from upstream `origin/main` @ `6a9109d86`, 2026-08-03; the
  #47808 branch may lag or lead). If a coordinate moved, note the new file:line
  before proceeding — a moved gate changes which hypothesis a failure rejects.

## 1. Capability precheck (on the rig, before serving)

The hypothesis chain for "runs on H100" and where each link is gated:

| link | coordinate (upstream 8/03) | pass condition |
|---|---|---|
| FA backend admits SM90 | `vllm/v1/attention/backends/flash_attn.py:201-202` `supports_compute_capability` → `capability >= DeviceCapability(8, 0)` | Hopper (9,0) ≥ (8,0) ✓ |
| **FA3 supports ragged qlen>1 (spec-decode) full CG** | `flash_attn.py:352-356` `_cudagraph_support = ALWAYS if get_flash_attn_version() == 3 else UNIFORM_BATCH` | `get_flash_attn_version() == 3` |
| adaptive verification scheduler | `adaptive_verification.py` (backend-agnostic, no SM ref) | n/a — no gate |
| (MLA-only, N/A for dense Gemma) | `mla/indexer.py` `use_fp4_indexer_cache` assert | keep the flag OFF |

Precheck command:

```bash
# NOTE: correct path is vllm.v1.attention.backends.fa_utils (verified on the rig
# 2026-08-04; attention lives under vllm/v1/attention/ now). Returned 3 on H100.
python -c "from vllm.v1.attention.backends.fa_utils import get_flash_attn_version; \
print('flash_attn_version =', get_flash_attn_version())"
# Expect: 3.  If 2 -> FA3 not built on this rig; _cudagraph_support drops to
# UNIFORM_BATCH and the ragged spec-decode + full-CG path is not available.
# Rebuild/enable FA3 before the smoke, or the failure is a rig artifact not a
# code gate.
```

## 2. Launch

Target `google/gemma-4-12B-it` (dense — the DSpark hybrid V2-maturity issue is a
separate track), drafter `deepseek-ai/dspark_gemma4_12b_block7`.

```bash
export HF_TOKEN=...        # authenticated pull (wall time is billed)
vllm serve google/gemma-4-12B-it \
  --speculative-config '{"method": "<dspark method name>", \
     "model": "deepseek-ai/dspark_gemma4_12b_block7", \
     "num_speculative_tokens": <K>}' \
  --seed 980406
```

> The exact `--speculative-config` schema (method name, the
> `confidence_ema_alpha` / adaptive-verification knobs) is **branch-specific**;
> confirm it against the #47808 branch's own tests/examples at rental time. Do
> not guess the flag names into a public artifact. What we know: the confidence
> EMA knob is `confidence_ema_alpha` (default 0.8), and the scheduler is
> capacity-realloc / adaptive.

## 3. Success criteria (pre-registered)

SUCCESS requires all three:
1. **Server boots** — no crash through model load + first decode.
2. **Adaptive verification active** — the capacity-realloc / adaptive-verification
   code path logs as engaged (not silently falling back to fixed-K).
3. **AR / AL in a sane band** — acceptance rate and mean accepted length are
   non-degenerate. Reference: fank's base-DSpark Hopper field report was
   acceptance ~57%, mean AL ~3.86 on 2×H100 NVL sm_90. The DSpark-Gemma head will
   differ, but AR≈0 or AL≈1 means the draft head is not engaging.

## 4. FAILURE protocol (capture, don't infer)

If any criterion fails, **capture the exact assert / traceback verbatim** — that
is the alternative data for the deferred B comment (a precise "SM90 is blocked at
X" is itself a contribution). Map the failure to the rejected hypothesis:

- Dies with `get_flash_attn_version() != 3` / FA2 path → **rig artifact** (FA3 not
  built), not a code gate. Rebuild FA3 and retry.
- Dies at a `supports_compute_capability` / capability assert → unexpected;
  capture which backend and the required capability. This would reject the
  "FA admits SM90" link.
- Dies inside `adaptive_verification.py` at a runtime assert → the
  backend-agnostic assumption is wrong; capture the file:line. Highest-value
  failure to record.
- Dies at `mla/indexer.py use_fp4_indexer_cache` → FP4 cache was enabled by
  accident (dense Gemma should not hit MLA at all); set it off.

## 5. On SUCCESS → the B comment (gated)

Collect: AR/AL numbers, the head SHA tested, the one flag/config needed, and the
precheck output (`get_flash_attn_version() == 3`). Draft the #47808 enablement
comment ("your branch runs on H100/SM90 — numbers + the flag"). Post only after
[[feedback-oss-pr-comment-timing]] 3-check + [[feedback-public-artifact-verification-gate]]
(live re-fetch of the branch, verbatim quotes, no measurement-free claims).

Credential-scan any logs/artifacts before they leave the rig.
