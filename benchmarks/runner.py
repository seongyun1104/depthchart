from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from benchmarks.metrics import EngineSnapshot, RequestMetric, RunResult, write_parquet
from benchmarks.sweep_config import SweepConfig, load
from lmcache.workload_gen import HitRateController, Prompt
from scheduler.vllm_runner import VLLMServer
from speculative.adapter import SpecConfig, for_exaone_45, for_gemma_4


async def _send_one(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: Prompt,
    req_id: str,
    hf_id: str,
    max_tokens: int,
) -> RequestMetric:
    body = {
        "model": hf_id,
        "messages": [{"role": "user", "content": prompt.text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
    }
    arrival_ts = time.time()
    first_token_ts = 0.0
    last_token_ts = 0.0
    prompt_tokens = prompt.prompt_tokens
    completion_tokens = 0

    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=None,
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            now = time.time()
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if first_token_ts == 0.0:
                        first_token_ts = now
                    last_token_ts = now
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

    if first_token_ts == 0.0:
        first_token_ts = arrival_ts
    if last_token_ts == 0.0:
        last_token_ts = first_token_ts

    return RequestMetric(
        request_id=req_id,
        arrival_ts=arrival_ts,
        first_token_ts=first_token_ts,
        last_token_ts=last_token_ts,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _parse_metric(text: str, name: str) -> float:
    total = 0.0
    matched = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head = line.split("{", 1)[0].split(" ", 1)[0]
        if head != name:
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
            matched = True
        except (ValueError, IndexError):
            continue
    return total if matched else 0.0


async def _snapshot(
    client: httpx.AsyncClient,
    base_url: str,
    controller: HitRateController,
) -> EngineSnapshot:
    r = await client.get(f"{base_url}/metrics")
    r.raise_for_status()
    text = r.text
    return EngineSnapshot(
        ts=time.time(),
        gpu_compute_util=0.0,
        hbm_bw_util=0.0,
        running_requests=int(_parse_metric(text, "vllm:num_requests_running")),
        queued_requests=int(_parse_metric(text, "vllm:num_requests_waiting")),
        workload_hit_rate=controller.realized_hit_rate,
        prefix_cache_queries_total=_parse_metric(text, "vllm:prefix_cache_queries_total"),
        prefix_cache_hits_total=_parse_metric(text, "vllm:prefix_cache_hits_total"),
        spec_num_drafts_total=_parse_metric(text, "vllm:spec_decode_num_drafts_total"),
        spec_num_draft_tokens_total=_parse_metric(text, "vllm:spec_decode_num_draft_tokens_total"),
        spec_num_accepted_tokens_total=_parse_metric(text, "vllm:spec_decode_num_accepted_tokens_total"),
        kv_cache_usage_perc=_parse_metric(text, "vllm:kv_cache_usage_perc"),
    )


async def _worker(
    client: httpx.AsyncClient,
    base_url: str,
    hf_id: str,
    max_tokens: int,
    controller: HitRateController,
    deadline_ts: float,
    collect_after_ts: float,
    collector: list[RequestMetric],
) -> None:
    while time.time() < deadline_ts:
        prompt = controller.next()
        req_id = uuid.uuid4().hex[:12]
        try:
            metric = await _send_one(client, base_url, prompt, req_id, hf_id, max_tokens)
        except Exception:
            continue
        if metric.arrival_ts >= collect_after_ts:
            collector.append(metric)


async def _poller(
    base_url: str,
    controller: HitRateController,
    snapshots: list[EngineSnapshot],
    stop_event: asyncio.Event,
    interval_s: float = 5.0,
) -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        while not stop_event.is_set():
            try:
                snap = await _snapshot(client, base_url, controller)
                snapshots.append(snap)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass


async def run_one(
    cfg: SweepConfig,
    batch_size: int,
    hit_rate: float,
    spec_k: int,
    spec_method: str,
) -> RunResult:
    if spec_method == "off" or spec_k == 0:
        spec = SpecConfig(method="off", num_steps=0)
    elif cfg.model.draft_model:
        spec = for_gemma_4(spec_k, cfg.model.draft_model)
    else:
        spec = for_exaone_45(spec_k, spec_method)  # type: ignore[arg-type]

    lmcache_cfg: Path | None = None
    if cfg.engine.enable_lmcache and cfg.engine.lmcache_config:
        lmcache_cfg = Path(cfg.engine.lmcache_config)

    server = VLLMServer(
        hf_id=cfg.model.hf_id,
        port=cfg.engine.port,
        lmcache_config=lmcache_cfg,
        spec=spec,
        quantization=cfg.model.quantization,
        max_num_batched_tokens=cfg.engine.chunked_prefill_size,
        gpu_memory_utilization=cfg.engine.mem_fraction_static,
        tp_size=cfg.engine.tp_size,
        max_model_len=cfg.model.max_context,
        reasoning_parser=cfg.model.reasoning_parser,
        tool_call_parser=cfg.model.tool_call_parser,
        kv_cache_dtype=cfg.engine.kv_cache_dtype,
    )

    async with server as handle:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id, trust_remote_code=True)
        controller = HitRateController(
            target_hit_rate=hit_rate,
            prompt_tokens=cfg.workload.prompt_tokens,
            shared_prefix_tokens=cfg.workload.prompt_tokens // 2,
            tokenizer=tokenizer,
        )
        requests: list[RequestMetric] = []
        snapshots: list[EngineSnapshot] = []
        stop_event = asyncio.Event()

        started = time.time()
        warmup_end = started + cfg.workload.warmup_s
        deadline = warmup_end + cfg.workload.duration_s

        poll_task = asyncio.create_task(
            _poller(handle.base_url, controller, snapshots, stop_event)
        )

        async with httpx.AsyncClient(timeout=None) as client:
            workers = [
                asyncio.create_task(_worker(
                    client, handle.base_url, cfg.model.hf_id,
                    cfg.workload.completion_tokens, controller,
                    deadline, warmup_end, requests,
                ))
                for _ in range(batch_size)
            ]
            await asyncio.gather(*workers, return_exceptions=True)

        stop_event.set()
        await poll_task
        ended = time.time()

    return RunResult(
        run_id=uuid.uuid4().hex[:12],
        batch_size=batch_size,
        hit_rate_target=hit_rate,
        spec_k=spec_k,
        spec_method=spec_method,
        requests=requests,
        engine_snapshots=snapshots,
        started_ts=warmup_end,
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
