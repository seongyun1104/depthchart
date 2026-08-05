# RUNBOOK — graph-mode tax decomposition (execute-only)

Companion to `PREREGISTRATION.md` (v3) and `master_bench_tax.py`. Everything
below is prepared at desk; the rental executes only. Wall-time ~3-4 h.

## Preflight (done at desk, 2026-08-05)

- **#49652**: OPEN, head `fd355781f71e` == pinned SHA. Branch
  `fix/mrv2-mtp-dsd-cudagraph-capture`. Use the pinned SHA regardless of drift.
- **DSD schedule API**: `num_speculative_tokens_per_batch_size` (3-tuple) is on
  upstream main and on the #49652 branch (`config/speculative.py:179`) — the
  dsd-K0 arm is not fork-dependent.
- **Models** (HF access confirmed, gated=False): target
  `prithivMLmods/gemma-4-31B-it-qat-FP8`, drafter
  `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`.
- **GPU**: H100 **NVL 94GB** required (matches #49986; 80GB PCIe changes the KV
  pool and voids spot check S). Offers seen at $2.16-2.80/hr, reliability 0.99.
- **#49986 config** restored verbatim from `pr49986_runbook/RUNBOOK.md`; encoded
  in `master_bench_tax.py`.

## 0. Provision

Rent an H100 NVL 94GB (`vastai search offers 'gpu_name=H100_NVL num_gpus=1
rentable=true'`, pick reliability >=0.99). SSH in. Confirm GPU/driver/python.

## 1. Environment + install (#49652 branch)

Recipe adapted from `pr49986_runbook/RUNBOOK.md` (proven), fork swapped for the
maintainer branch:

```bash
source /venv/main/bin/activate    # or the image's venv
export PYTHONHASHSEED=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_TOKEN=<token>            # `hf auth token`, account jamesyun
export HF_HUB_ENABLE_HFTRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

uv pip install vllm pytest tblib setuptools_scm setuptools-rust cmake ninja \
    packaging wheel jinja2 pandas pyarrow hf_transfer

cd /workspace
# full history, lazy blobs — a shallow clone breaks the PR-head fetch (P2' lesson)
git clone --filter=blob:none https://github.com/vllm-project/vllm.git
cd vllm
git fetch origin pull/49652/head
git checkout fd355781f71e          # pinned; the script records actual HEAD
uv pip uninstall vllm
VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation
uv pip install --upgrade quack-kernels
uv pip install nvidia-cutlass-dsl==4.6.0
python3 -c 'import vllm; print("vllm OK", vllm.__version__)'
```

## 2. Model download

```bash
hf download prithivMLmods/gemma-4-31B-it-qat-FP8 --local-dir /root/models/target
hf download google/gemma-4-31B-it-qat-q4_0-unquantized-assistant \
    --local-dir /root/models/draft
```

## 3. Run — arms A/B/C/D on the branch build

`master_bench_tax.py` handles launch / health-poll / burn / 3 warm + 3 measure /
metrics snapshot / kill. It sets `VLLM_USE_V2_MODEL_RUNNER` per arm (the only
lever). **Do not** pass `--kv-cache-dtype fp8` (FlashInfer SM90 guard, #48495).

```bash
cd /workspace/depthchart/tax_decomposition   # or scp this file + the script up
export RESULTS_DIR=/root/results

# Gate B + Gate T first, cheap fail-fast:
python master_bench_tax.py v1_no_spec v1_dsd_k0   # A, B  -> Gate T
python master_bench_tax.py v2_no_spec v2_dsd_k0   # C, D  -> Gate B (auto-captured)
```

- **Gate B** (V2 arms): if `v2_no_spec`/`v2_dsd_k0` fail `/health` in 600s, the
  server log `/tmp/server_v2_*.log` is the primary #49652 deliverable (capture
  the #48494 assert / other blocker). `gate_b_{arm}.json` records
  assert-seen / full-cudagraph-seen. On hard fail: fall back to the proxy
  (dense + EAGLE-3) per pre-reg §3, or stop and report.
- **Gate T** (from A vs B, ctx 400, TPOT p50): >=10% -> proceed; <5% -> the
  published -31% no longer reproduces on this build = **#49986 correction
  event**, do not attribute a vanished tax (pre-reg §6).

## 4. Spot check S — rebuild main, run `spot`

```bash
cd /workspace/vllm
git checkout origin/main           # the script records the SHA per arm
# editable install: a checkout alone updates the Python source. Reinstall ONLY
# if import fails (P2' proved dfda6a2<->fbb2246 switched with no reinstall).
python -c 'import vllm' \
  || VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation
cd /workspace/depthchart/tax_decomposition
python master_bench_tax.py spot    # main build, V1, dsd-K0, == #49986 point
                                   # (spot also runs ctx4000; only ctx400 feeds S)
```

Confirms branch-V1 (arm B) ≈ main-V1 (spot) ≈ published #49986 -31%. If
S != B beyond noise, the branch altered V1 behavior -> report, re-scope.

## 5. Dump + destroy

```bash
tar czf /tmp/tax_$(date +%Y-%m-%d).tar.gz /root/results /tmp/server_*.log
scp -P <port> root@<host>:/tmp/tax_*.tar.gz ./
# then destroy the instance (Vast workspace is ephemeral)
```

Extract into `raw/` here; push to depthchart with build SHAs recorded.

## 6. Analysis (pre-reg §4/§6, frozen)

Per ctx, from TPOT p50 (measure seeds only, warmup discarded):

- `Tax(V1) = (B - A) / A`, `Tax(V2) = (D - C) / C`
- **graph-mode component = Tax(V1) - Tax(V2)**; **residual = Tax(V2)**
- Baseline guard: if A(no_spec) vs C(no_spec) differ beyond noise, deltas are
  confounded -> report runner-baseline-dependent, do not attribute.
- Map to the frozen §6 matrix (V2<V1 / V1≈V2 / **V2>V1 = MRV2 caution** /
  Gate-T-null = #49986 correction / Gate-B-fail = #49652 report).
- **Tax is the K=0 arm only.** Do not write results before the run.
- **drafts sanity (dsd-K0 arms):** `drafts_total` should be small but non-zero
  (~hundreds) — the ramp region (batch < 129) fires the K=3 tier. Exactly 0
  means the schedule was not applied; a large value means wrong-tier landing.
  `aggregate_tax.py` prints it per cell.
- Run `RESULTS_DIR=./raw python aggregate_tax.py` — computes everything above;
  no eyeballing six JSONs under time pressure.

## 7. Deliverables (priority order)

1. **#49652 comment** — Gate B result (pass or fail). Highest-certainty
   contribution on the board; third-party verification of a small under-review
   fix, unlike the #47808 thread.
2. **#49986 update** — the decomposition (whichever §6 branch fired).
3. **depthchart** — raw + this runbook + summary.
4. (if applicable) **#48944 note** — residual = K=0 tier cost after V2.
