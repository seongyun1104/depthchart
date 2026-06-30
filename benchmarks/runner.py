from __future__ import annotations

import argparse
import asyncio
import itertools
import time
import uuid
from pathlib import Path

from benchmarks.metrics import RunResult, write_parquet
from benchmarks.sweep_config import SweepConfig, load


async def run_one(cfg: SweepConfig, batch_size: int, hit_rate: float,
                  spec_k: int, spec_method: str) -> RunResult:
    # TODO(t16-impl): launch SGLang server with (spec_method, spec_k, lmcache),
    # generate workload with target hit_rate, drive concurrency=batch_size,
    # collect RequestMetric per request + EngineSnapshot periodically.
    started = time.time()
    await asyncio.sleep(0)
    ended = time.time()
    return RunResult(
        run_id=uuid.uuid4().hex[:12],
        batch_size=batch_size,
        hit_rate_target=hit_rate,
        spec_k=spec_k,
        spec_method=spec_method,
        requests=[],
        engine_snapshots=[],
        started_ts=started,
        ended_ts=ended,
    )


async def sweep(cfg: SweepConfig) -> list[RunResult]:
    results: list[RunResult] = []
    grid = itertools.product(cfg.axes.batch_sizes, cfg.axes.hit_rates, cfg.axes.spec_k)
    for b, h, k in grid:
        method = "off" if k == 0 else cfg.axes.spec_method
        result = await run_one(cfg, b, h, k, method)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    args = parser.parse_args()

    cfg = load(args.config)
    results = asyncio.run(sweep(cfg))
    out = write_parquet(results, args.out)
    print(f"wrote {len(results)} runs to {out}")


if __name__ == "__main__":
    main()
