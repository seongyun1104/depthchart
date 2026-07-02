from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


SpecMethod = Literal["off", "mtp", "nextn", "eagle", "eagle3", "fastmtp", "dflash"]
Quantization = Literal["none", "fp8", "awq", "gptq",
                       "modelopt_fp4", "compressed-tensors"]


@dataclass(frozen=True)
class SweepAxes:
    batch_sizes: tuple[int, ...] = (1, 4, 16, 32, 64, 128)
    hit_rates: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
    spec_k: tuple[int, ...] = (0, 1, 2, 3)
    spec_method: SpecMethod = "mtp"
    eagle_topk: int = 1
    num_draft_tokens: int = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    spec_methods: tuple[SpecMethod, ...]
    max_context: int
    quantization: Quantization = "none"
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    draft_model: str | None = None


@dataclass(frozen=True)
class EngineSpec:
    host: str = "127.0.0.1"
    port: int = 8000
    enable_lmcache: bool = True
    lmcache_config: str | None = None
    chunked_prefill_size: int = 8192
    mem_fraction_static: float = 0.85
    tp_size: int = 1
    gpu_vram_gb: int = 96
    kv_cache_dtype: str | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    prompt_tokens: int = 2048
    completion_tokens: int = 256
    concurrency: int = 128
    duration_s: int = 60
    warmup_s: int = 10


@dataclass(frozen=True)
class SweepConfig:
    axes: SweepAxes = field(default_factory=SweepAxes)
    model: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            name="exaone-4.5-33b-fp8",
            hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
            spec_methods=("mtp", "eagle"),
            max_context=32768,
            quantization="fp8",
            reasoning_parser="qwen3",
            tool_call_parser="hermes",
        )
    )
    engine: EngineSpec = field(default_factory=EngineSpec)
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    results_dir: Path = field(default_factory=lambda: Path("results"))


def load(path: str | Path) -> SweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return _from_dict(raw)


def _from_dict(raw: dict) -> SweepConfig:
    axes_raw = dict(raw.get("axes", {}))
    for k in ("batch_sizes", "hit_rates", "spec_k"):
        if k in axes_raw:
            axes_raw[k] = tuple(axes_raw[k])
    axes = SweepAxes(**axes_raw)

    model_raw = dict(raw["model"])
    if "spec_methods" in model_raw:
        model_raw["spec_methods"] = tuple(model_raw["spec_methods"])
    model = ModelSpec(**model_raw)

    engine = EngineSpec(**raw.get("engine", {}))
    workload = WorkloadSpec(**raw.get("workload", {}))
    results_dir = Path(raw.get("results_dir", "results"))
    return SweepConfig(axes=axes, model=model, engine=engine,
                       workload=workload, results_dir=results_dir)
