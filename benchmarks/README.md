# benchmarks

Sweep harness for (B × hit_rate × K) on a single GPU.

```
sweep_config.py      strongly-typed config + YAML loader
configs/             YAML sweep specs (one per model)
runner.py            sweep driver
metrics.py           RequestMetric / EngineSnapshot / RunResult + parquet writer
```

## Run

```
uv run python -m benchmarks.runner --config benchmarks/configs/exaone_4_5_33b_fp8.yaml
```

## Status

Skeleton only. `run_one()` is a no-op stub. Engine lifecycle, workload
driver, and metric collection plug in via separate PRs from this directory's
sibling modules (`lmcache/`, `speculative/`).

Engine lifecycle owner: `scheduler/sglang_runner.py` (TBD)
Workload owner: `lmcache/workload_gen.py`
Spec hook owner: `speculative/adapter.py`
