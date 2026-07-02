from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RequestMetric:
    request_id: str
    arrival_ts: float
    first_token_ts: float
    last_token_ts: float
    prompt_tokens: int
    completion_tokens: int
    accepted_tokens: int = 0
    drafted_tokens: int = 0
    cache_hit_tokens: int = 0

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_ts - self.arrival_ts) * 1000

    @property
    def itl_ms(self) -> float:
        gen = max(self.completion_tokens - 1, 1)
        return (self.last_token_ts - self.first_token_ts) * 1000 / gen

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0


@dataclass
class EngineSnapshot:
    ts: float
    gpu_compute_util: float
    hbm_bw_util: float
    running_requests: int
    queued_requests: int
    workload_hit_rate: float
    prefix_cache_queries_total: float = 0.0
    prefix_cache_hits_total: float = 0.0
    spec_num_drafts_total: float = 0.0
    spec_num_draft_tokens_total: float = 0.0
    spec_num_accepted_tokens_total: float = 0.0
    kv_cache_usage_perc: float = 0.0


@dataclass
class RunResult:
    run_id: str
    batch_size: int
    hit_rate_target: float
    spec_k: int
    spec_method: str
    requests: list[RequestMetric]
    engine_snapshots: list[EngineSnapshot]
    started_ts: float
    ended_ts: float

    def throughput_tok_s(self) -> float:
        total = sum(r.completion_tokens for r in self.requests)
        wall = max(self.ended_ts - self.started_ts, 1e-9)
        return total / wall

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for r in self.requests:
            rows.append({
                "run_id": self.run_id,
                "batch_size": self.batch_size,
                "hit_rate_target": self.hit_rate_target,
                "spec_k": self.spec_k,
                "spec_method": self.spec_method,
                **asdict(r),
                "ttft_ms": r.ttft_ms,
                "itl_ms": r.itl_ms,
                "acceptance_rate": r.acceptance_rate,
            })
        return rows


def write_parquet(results: list[RunResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for r in results for row in r.to_records()]
    df = pd.DataFrame(rows)
    path = out_dir / "requests.parquet"
    df.to_parquet(path, index=False)
    return path
