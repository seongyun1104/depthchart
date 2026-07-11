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
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0


@dataclass
class RunResult:
    run_id: str
    batch_size: int
    ctx_tokens: int
    hit_rate_target: float
    spec_k: int
    spec_method: str
    requests: list[RequestMetric]
    engine_snapshots: list[EngineSnapshot]
    started_ts: float
    ended_ts: float
    hit_source: str = "apc"
    drafter_loaded: bool = False
    spec_method_logged: str | None = None
    kv_pool_tokens: int | None = None
    enable_prefix_caching: bool = True
    enable_lmcache: bool = False

    def _window_snapshots(self) -> list[EngineSnapshot]:
        return [s for s in self.engine_snapshots
                if self.started_ts <= s.ts <= self.ended_ts]

    def throughput_tok_s(self) -> float:
        window = self._window_snapshots()
        if len(window) < 2:
            return 0.0
        first, last = window[0], window[-1]
        d_gen = max(last.generation_tokens_total - first.generation_tokens_total, 0.0)
        wall = max(last.ts - first.ts, 1e-9)
        return d_gen / wall

    def throughput_tok_s_client(self) -> float:
        total = sum(r.completion_tokens for r in self.requests)
        wall = max(self.ended_ts - self.started_ts, 1e-9)
        return total / wall

    def prefill_share(self) -> float:
        window = self._window_snapshots()
        if len(window) < 2:
            return 0.0
        first, last = window[0], window[-1]
        d_prompt = max(last.prompt_tokens_total - first.prompt_tokens_total, 0.0)
        d_gen = max(last.generation_tokens_total - first.generation_tokens_total, 0.0)
        denom = d_prompt + d_gen
        return d_prompt / denom if denom > 0 else 0.0

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        prefill_share = self.prefill_share()
        throughput_counter = self.throughput_tok_s()
        throughput_client = self.throughput_tok_s_client()
        for r in self.requests:
            rows.append({
                "run_id": self.run_id,
                "batch_size": self.batch_size,
                "ctx_tokens": self.ctx_tokens,
                "hit_rate_target": self.hit_rate_target,
                "spec_k": self.spec_k,
                "spec_method": self.spec_method,
                "hit_source": self.hit_source,
                "drafter_loaded": self.drafter_loaded,
                "spec_method_logged": self.spec_method_logged,
                "kv_pool_tokens": self.kv_pool_tokens,
                "enable_prefix_caching": self.enable_prefix_caching,
                "enable_lmcache": self.enable_lmcache,
                "prefill_share": prefill_share,
                "throughput_tok_s_counter": throughput_counter,
                "throughput_tok_s_client": throughput_client,
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
