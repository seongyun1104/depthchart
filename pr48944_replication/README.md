# PR #48944 replication package (2026-07-27)

Full repro material for the measurement decomposition in [vLLM PR #48944 comment #5091663057](https://github.com/vllm-project/vllm/pull/48944#issuecomment-5091663057).

## Contents

- `master_bench.py` — orchestrator: launches vLLM server per arm, wipes autotune cache, runs cold-start burn + `prefix_repetition` sweep with 3 warmup + 3 measure runs, snapshots Prometheus deltas per cell.
- `aggregate_balanced.json` — T1 + T2 + position-balanced (T1+T2)/2 aggregate (the source of Table 1 in the PR comment).
- `aggregate_phase4.json` — Trial 1 per-cell measurements.
- `aggregate_eager.json` — Eager control run (discarded, mml=4096 conditions did not match §2; retained for transparency).
- `aggregate_recheck.json` — C_prime ctx=400 c=192 recheck (preempt contamination reproof).

## Environment (checked)

- Hardware: **H100 NVL 96GB**, driver 580.82.09, CUDA 13.0.88
- vLLM: `feat/dsd-2d-ctx-schedule` @ `c5d967c23fbedcd4085ea3e90ad4c806b40c55be` (PR #48944 head)
- Target: `prithivMLmods/gemma-4-31B-it-qat-FP8`
- Draft: `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`

## Install (Vast.ai or equivalent, ~15 min)

```bash
source /venv/main/bin/activate
export HF_TOKEN=<your token>
export HF_HUB_ENABLE_HFTRANSFER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONHASHSEED=0

# Base deps
uv pip install vllm pytest tblib setuptools_scm setuptools-rust cmake ninja packaging wheel jinja2 pandas pyarrow hf_transfer

# PR head editable install
mkdir -p ~/work && cd ~/work
git clone --depth 5 --branch feat/dsd-2d-ctx-schedule https://github.com/seongyun1104/vllm.git
cd vllm
pip uninstall -y vllm
VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation

# Ensure quack + cutlass compatible versions (7/17 recipe)
pip install --upgrade quack-kernels
pip install nvidia-cutlass-dsl==4.6.0

# Model pull
mkdir -p /root/models
hf download prithivMLmods/gemma-4-31B-it-qat-FP8 --local-dir /root/models/target
hf download google/gemma-4-31B-it-qat-q4_0-unquantized-assistant --local-dir /root/models/draft
```

## Run full measurement pass (Phase 4a + 4b, ~90 min)

```bash
# Trial 1 (order: C→N→S→A). Results go to /root/results/phase4/
export RESULTS_DIR=/root/results
python3 master_bench.py \
  'C_prime:phase4' 'no_spec:phase4' 'static_k3:phase4' 'A_prime:phase4'

# Trial 2 (reverse order: A→S→N→C, position-balanced). Separate dir.
export RESULTS_DIR=/root/results_t2
python3 master_bench.py \
  'A_prime:phase4' 'static_k3:phase4' 'no_spec:phase4' 'C_prime:phase4'

# Optional: preempt-free recheck of C_prime ctx=400 at c=192
export RESULTS_DIR=/root/results
python3 master_bench.py 'C_prime_recheck_c192:phase4'
```

## Schedule definitions (SCHEDULES dict in master_bench.py)

- `A_prime` = 3-item batch-only: `[[1,64,3], [65,128,1], [129,512,0]]`
- `C_prime` = 6-cell 2D with mid tier long-ctx K=1 preserved (single-cell isolation):
  ```
  [[1,64,1,768,3], [1,64,769,32768,3],
   [65,128,1,768,1], [65,128,769,32768,1],
   [129,512,1,768,0], [129,512,769,32768,3]]
  ```
- `no_spec` = no `speculative_config`
- `static_k3` = `[[1,512,3]]`

`ctx_agg` defaults to `mean` for 2D schedules.

## Bench command (fired per (arm, ctx) cell by master_bench.py)

```bash
vllm bench serve --model /root/models/target --port 8000 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len <CTX>     # 400, 900, 1900, 4000
  --prefix-repetition-suffix-len 96 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 100 \
  --num-prompts 256 --max-concurrency 256 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --save-result --result-dir <out> --result-filename <run>.json
```

Between each server launch: `rm -rf ~/.cache/vllm/flashinfer_autotune_cache/`, then 2 rounds of a short random burn (c=64, ISL=512, OSL=64) before real measurements start. 3 warmup runs are discarded, 3 measure runs are saved with Prometheus counter deltas (drafts, accepted, preempts, prefix_cache_hits) snapshotted around each.

## References

- MagicDec (Sadhukhan et al., 2024): [arXiv:2408.11049](https://arxiv.org/abs/2408.11049) — the prior formalization of "K as function of sequence length" that this work extends
- Original vLLM DSD API: [vllm-project/vllm#32374](https://github.com/vllm-project/vllm/pull/32374) (ekagra-ranjan)
- This RFC: [vllm-project/vllm#48627](https://github.com/vllm-project/vllm/issues/48627)
- This PR: [vllm-project/vllm#48944](https://github.com/vllm-project/vllm/pull/48944)
- Follow-up (DSD baseline tax): [vllm-project/vllm#49986](https://github.com/vllm-project/vllm/issues/49986)
- Companion SGLang PR: [sgl-project/sglang#31716](https://github.com/sgl-project/sglang/pull/31716)
