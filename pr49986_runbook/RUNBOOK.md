# RUNBOOK — P2' K=0 fast-path measurement

Session runbook that produced the raw artefacts in `raw/` and the summary
table in the [2026-08-03 follow-up comment on
#49986](https://github.com/vllm-project/vllm/issues/49986#issuecomment-5162771381).

Wall-time budget: about 90 minutes on a fresh H100 rental (Vast.ai H100 NVL
class) with a warm HF cache. Install + model pull dominate the first
half; the three measurement blocks fit inside the second half.

## Fork state

- Repo: `github.com/seongyun1104/vllm`
- Branch: `feat/dsd-k0-fastpath`
- Base: upstream `vllm-project/vllm@437e0b7` (2026-07-30)
- Three commits on the branch (order-independent for the on/off diff-clean
  comparison below):
  - `dfda6a2df` `[Spec Decode] Declare cross-model KV-sharing capability flag`
  - `5a398a660` `[Spec Decode] Skip drafter forward at K=0 for shared-KV proposers`
  - `fbb2246c6` `[Spec Decode] Test K=0 fast-path predicate`

## Two SHAs for the on/off comparison

The on/off states are compared by checking out different commits rather than
by toggling an environment variable. This keeps the upstream diff clean and
prevents runtime feature flags from leaking into the measurement.

| State | SHA | Meaning |
|---|---|---|
| **on**  | `fbb2246c6` (HEAD)     | Guard active; at K=0 the drafter forward is skipped for shared-KV proposers. |
| **off** | `dfda6a2df` (guard parent) | Capability flag present but no guard call site; K=0 falls through the existing empty-return path in `propose_draft_token_ids`. |

`dfda6a2df` sits above upstream `main` because the capability-flag commit
defines `HAS_CROSS_MODEL_KV_SHARING` on the Gemma4 MTP drafter. Since neither
server calls the guard, the on/off delta isolates the guard code exactly —
same code surface, no environment-variable toggle needed.

Both servers must be launched with a full cold-start burn between the
checkout and the measurement runs. See "Cold-start burn" below.

## Model and config

- **Target**: `prithivMLmods/gemma-4-31B-it-qat-FP8` (community re-upload of
  the Google Gemma-4-31B-it QAT FP8 weights)
- **Drafter (MTP)**: `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`
  (Google-gated; requires an HF token with read access to the Google org)
- **DSD schedule (batch-only 3-tuple)**:
  `[[1,64,3],[65,128,0],[129,512,0]]` — high-batch (bs>=65) drops to K=0
  so the ctx=400 workload lands in the tier where the drafter-forward
  question actually matters.
- **Engine CLI**:
  `--gpu-memory-utilization 0.90 --max-model-len 8192`
  (add `--enforce-eager` only when eager-mode arms are separately needed)
- **Do not** pass `--kv-cache-dtype fp8` on this stack. The drafter's
  attention backend auto-selector picks FlashInfer, and vLLM's FlashInfer
  guard rejects SM90 + sliding-window drafter combinations at engine-core
  init. This is the failure mode in `vllm-project/vllm#48495`.

Model download uses the new HF CLI (`hf download`, not `huggingface-cli`):

```bash
hf download prithivMLmods/gemma-4-31B-it-qat-FP8 --local-dir /root/models/target
hf download google/gemma-4-31B-it-qat-q4_0-unquantized-assistant --local-dir /root/models/draft
```

## Environment

Export before installing and before every `vllm serve`:

```bash
source /venv/main/bin/activate
export PYTHONHASHSEED=0                  # determinism norm for greedy diff check
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_TOKEN=<your-token>             # required for the drafter (Google-gated)
export HF_HUB_ENABLE_HFTRANSFER=1        # faster model pulls
export VLLM_LOGGING_LEVEL=INFO
```

`PYTHONHASHSEED=0` must be exported in the shell that runs `vllm serve` and
the shell that runs the greedy-diff check. Losing it silently breaks the
byte-identical determinism assertion.

## Install recipe

```bash
# uv pip install vllm pulls torch, flashinfer, and all runtime deps together,
# so no --no-deps, no manual flashinfer pin, no _C shim are needed.
uv pip install vllm pytest tblib setuptools_scm setuptools-rust cmake ninja \
    packaging wheel jinja2 pandas pyarrow hf_transfer

# Clone fork + editable install so checkouts between the two SHAs recompile
# the guard cheaply.
cd /workspace
git clone --depth 20 --branch feat/dsd-k0-fastpath \
    https://github.com/seongyun1104/vllm.git
cd vllm
pip uninstall -y vllm
VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation

# Version corrections that the uv-resolved matrix sometimes misses:
pip install --upgrade quack-kernels        # >= 0.6.1 for the new cutlass API
pip install nvidia-cutlass-dsl==4.6.0      # match vLLM's declared version

# Smoke:
python3 -c 'import vllm; from vllm import LLM; print("vllm OK", vllm.__version__)'
```

## Cold-start burn (mandatory)

For the first 10-30 minutes after startup, tok/s is depressed by 20-40 %
because CUDA graph capture and FlashInfer JIT autotune have not yet settled.
Every measurement below assumes the burn has completed:

1. Start the server.
2. Fire two small `vllm bench serve` rounds of `--num-prompts 64
   --max-concurrency 64` against a synthetic prompt file to force capture
   of every batch size in the schedule.
3. Sleep 30 s, then start the real measurement.

Repeat the burn every time the server is restarted (including between the
on and off checkouts).

## Session sequence

Three measurement blocks. (a) is the K=0 fast-path on/off comparison at
ctx=400 that the follow-up comment §1 table reports. (b) verifies the same
effect on the batch-only 3-tuple schedule (the pre-`#48944` DSD surface).
(c) is the 15-minute V2 model runner escape-hatch probe from §3.

### (a) Fast-path on/off at ctx=400

1. `git checkout dfda6a2df` (off).
2. Launch `vllm serve`:

    ```bash
    vllm serve /root/models/target \
      --port 8000 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 8192 \
      --speculative-config '{"model": "/root/models/draft", "num_speculative_tokens": 3, "num_speculative_tokens_per_batch_size": [[1,64,3],[65,128,0],[129,512,0]]}'
    ```

    (Do not add `--kv-cache-dtype fp8`.)

3. Cold-start burn (see above), then `sleep 30`.
4. Bench: `--num-prompts 1024 --max-concurrency 256 --ignore-eos
   --dataset-name prefix_repetition --prefix-repetition-prefix-len 400
   --prefix-repetition-suffix-len 96 --prefix-repetition-num-prefixes 1
   --prefix-repetition-output-len 100 --percentile-metrics ttft,tpot,itl`.
   Discard the first three warmup outputs, keep the next three measure
   outputs.
5. Greedy determinism check: run the same server against 5-10 fixed prompts
   with `temperature=0`, `seed=42`; save the outputs.
6. Stop the server. `git checkout fbb2246c6` (on).
7. Repeat steps 2-5 with the same parameters.
8. Compare the greedy outputs from step 5 across on and off. They must be
   byte-identical when both servers are launched cleanly; if they diverge,
   the most likely cause is a container-restart carryover in the CUDA
   context, not the guard.

### (b) Batch-only schedule (independence check)

Same procedure as (a), but with the DSD schedule
`[[1,64,3],[65,128,1],[129,512,0]]` (the pre-`#48944` batch-only DSD form).
This confirms the fast-path effect (or, as it turned out here, its absence
above the noise floor) does not depend on the 5-tuple ctx-axis schema.

### (c) V2 model runner escape hatch (§3 of the comment)

1. `git checkout fbb2246c6`, add `VLLM_USE_V2_MODEL_RUNNER=1` to the shell.
2. Launch the same `vllm serve` command as in (a).
3. Three checks, in order:
   - The server must reach the "ready" state without a runtime assertion
     from `#48494` (`InputBatch.make_dummy` full-CG capture assert).
   - Server logs must contain "FULL cudagraph" for the decode batches,
     confirming the CG mode is preserved. Search for the string in
     `raw/logs/vllm_v2.log` as an example.
   - Run the same greedy determinism prompts as in (a); confirm the output
     matches the on-branch greedy result byte-for-byte.

## After the session

- Copy the results directory tree out of the rental container before
  destroying the instance (Vast.ai `workspace_is_volume=false` means the
  workspace disappears at destroy). One-liner:

    ```bash
    tar czf /tmp/p2prime_$(date +%Y-%m-%d).tar.gz /root/results /tmp/bench_*.json /tmp/greedy_*.json /tmp/vllm_*.log
    scp -P <port> root@<host>:/tmp/p2prime_*.tar.gz ./
    ```

- The `raw/` layout in this directory is the extracted form of that
  tarball, with `root/results/{on,off}/` moved to
  `raw/measurements/{on,off}/`, the `/tmp/bench_*.json` and greedy JSONs
  moved to `raw/aggregate/`, and the server logs moved to `raw/logs/`.

## Wall-time budget

| Phase | Approx |
|---|---|
| Install + fork clone | 15 min |
| Model download (authenticated) | 20 min |
| Cold-start burn (twice, on and off) | 8 min |
| (a) fast-path on/off | 20 min |
| (b) batch-only schedule | 12 min |
| (c) V2 escape hatch | 15 min |
| Data dump + tarball out + destroy | 5 min |
