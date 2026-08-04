# pr49986_runbook — P2' K=0 fast-path exploration (parked)

Runbook and raw measurements backing the [2026-08-03 follow-up
comment](https://github.com/vllm-project/vllm/issues/49986#issuecomment-5162771381)
on [vllm-project/vllm#49986](https://github.com/vllm-project/vllm/issues/49986).

**Status: parked, not landed.** The K=0 drafter-forward skip predicate was
implemented, tested, and measured, but the measured effect at ctx=400 was
sub-noise (TPOT +0.71%, output tok/s +0.89%, both inside a run-to-run noise
floor of about 2–3 %). The three capability/predicate/test commits are kept
on the fork branch as reusable assets; no PR was opened on `vllm-project/vllm`.

This directory publishes the reproducibility artefacts referenced by that
follow-up comment so the numbers can be independently re-run and audited.

## What's here

```
pr49986_runbook/
├── README.md           this file
├── RUNBOOK.md          install recipe, cold-start burn, per-arm session order
└── raw/
    ├── measurements/
    │   ├── on/         K=0 fast-path guard active (fork HEAD fbb2246c6)
    │   │   warmup_{0,1,2}.json + measure_{0,1,2}.json
    │   └── off/        guard inactive, parent commit dfda6a2df
    │       warmup_{0,1,2}.json + measure_{0,1,2}.json
    ├── aggregate/
    │   ├── bench_off.json, bench_on.json    aggregate vllm bench serve output
    │   └── greedy_{off,on}[_seq].json       byte-identical determinism outputs
    └── logs/
        ├── vllm_off_final.log, vllm_on2.log server logs for the on/off runs
        └── vllm_v2.log                       VLLM_USE_V2_MODEL_RUNNER=1 escape-hatch
                                              probe (§3 of the follow-up comment)
```

Each `warmup_*.json` / `measure_*.json` is a per-run `vllm bench serve`
output (Prometheus-counter throughput, TPOT/TTFT percentiles, acceptance
counters). The warmups are the three discarded runs preceding each state's
three measurement runs.

## How each artefact maps to the comment

| Comment section | Backing files |
|---|---|
| §1 table (Mean TPOT / Median TPOT / Output tok/s / Acceptance % / Drafts, off vs on, 3-run mean of steady-state measures) | `raw/measurements/off/measure_{0,1,2}.json` and `raw/measurements/on/measure_{0,1,2}.json`; aggregates in `raw/aggregate/bench_{off,on}.json` |
| §1 "byte-identical off vs on when both servers are launched cleanly" | `raw/aggregate/greedy_{off,on}.json` and `raw/aggregate/greedy_{off,on}_seq.json` — same-prompt, `temperature=0`, `seed=42` greedy outputs across both branches |
| §2 tax decomposition, "drafter forward is not the culprit (≈ 0)" | Derived from §1 measurements; no additional file. |
| §3 V2 escape-hatch probe (`VLLM_USE_V2_MODEL_RUNNER=1`) | `raw/logs/vllm_v2.log` |

## Setup that produced these numbers

- Hardware: H100 NVL, driver 580.173.02, CUDA 13.0.
- Target: `prithivMLmods/gemma-4-31B-it-qat-FP8`.
- Drafter: `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`.
- Fork: `github.com/seongyun1104/vllm@feat/dsd-k0-fastpath`, base upstream
  `437e0b7`, HEAD `fbb2246c6`, precompiled binaries from upstream `0033211`.
- Workload: `vllm bench serve --dataset-name prefix_repetition
  --prefix-repetition-{prefix-len,suffix-len,num-prefixes,output-len} 400 96 1 100`,
  `--num-prompts 1024 --max-concurrency 256 --ignore-eos`.
- 3 warmup runs discarded, 3 measure runs kept per state. Autotune cache
  wiped between states; container restarted between the two servers to
  avoid CUDA-context carryover (the initial off/on divergence I saw before
  doing this cleanly was a container-restart confound, not the code diff).

Everything else — Python env, `PYTHONHASHSEED=0`, cold-start burn, the two
comparison SHAs (`dfda6a2df` for off, `fbb2246c6` for on) — is documented
in [`RUNBOOK.md`](./RUNBOOK.md).

## Reproducing

Independent replication follows `RUNBOOK.md` end-to-end on a comparable H100
rental (~90 minutes wall-time including install and model pull). No
proprietary datasets or credentials required beyond a Hugging Face token
for the Google-hosted drafter checkpoint.

## Related

- Companion PR (still open, receiving reviewer feedback separately):
  [`vllm-project/vllm#48944`](https://github.com/vllm-project/vllm/pull/48944)
- Adaptive verification track referenced in §2 of the follow-up comment
  (per-request budget rescheduling would raise K=0 frequency and change
  the arithmetic of parking this fast-path):
  [`vllm-project/vllm#47808`](https://github.com/vllm-project/vllm/pull/47808)
- Depthchart bench harness that produced `pr48944_replication/` and shares
  the same `master_bench.py` orchestrator conventions: see the sibling
  `pr48944_replication/` directory in this repo.
