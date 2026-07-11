from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


SpecMethod = Literal["off", "mtp", "nextn", "eagle", "eagle3", "fastmtp", "dflash"]
Quantization = Literal["none", "fp8", "awq", "gptq",
                       "modelopt_fp4", "compressed-tensors"]
HitSource = Literal["apc", "lmcache"]


@dataclass(frozen=True)
class SweepAxes:
    batch_sizes: tuple[int, ...] = (1, 4, 16, 32, 64, 128, 192, 256)
    ctx_tokens: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
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
    dsd_schedule: tuple[tuple[int, int, int], ...] | None = None


@dataclass(frozen=True)
class EngineSpec:
    host: str = "127.0.0.1"
    port: int = 8000
    lmcache_config: str | None = None
    chunked_prefill_size: int = 8192
    mem_fraction_static: float = 0.85
    tp_size: int = 1
    gpu_vram_gb: int = 96
    kv_cache_dtype: str | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    completion_tokens: int = 256
    duration_s: int = 120
    warmup_s: int = 30
    hit_rate: float = 0.0
    hit_source: HitSource = "apc"


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


_LEGACY_AXIS_KEYS = ("hit_rates", "hit_sources")
_LEGACY_WORKLOAD_KEYS = ("prompt_tokens", "concurrency")
_LEGACY_ENGINE_KEYS = ("enable_lmcache",)


def _from_dict(raw: dict) -> SweepConfig:
    axes_raw = dict(raw.get("axes", {}))
    legacy = [k for k in _LEGACY_AXIS_KEYS if k in axes_raw]
    if legacy:
        raise ValueError(
            f"axes.{','.join(legacy)} removed in P1.4 — hit_rate/hit_source demoted "
            f"to workload covariates. Move to workload.hit_rate and workload.hit_source."
        )
    for k in ("batch_sizes", "ctx_tokens", "spec_k"):
        if k in axes_raw:
            axes_raw[k] = tuple(axes_raw[k])
    axes = SweepAxes(**axes_raw)

    model_raw = dict(raw["model"])
    if "spec_methods" in model_raw:
        model_raw["spec_methods"] = tuple(model_raw["spec_methods"])
    if "dsd_schedule" in model_raw and model_raw["dsd_schedule"] is not None:
        model_raw["dsd_schedule"] = tuple(
            tuple(row) for row in model_raw["dsd_schedule"]
        )
    model = ModelSpec(**model_raw)

    engine_raw = dict(raw.get("engine", {}))
    legacy_eng = [k for k in _LEGACY_ENGINE_KEYS if k in engine_raw]
    if legacy_eng:
        raise ValueError(
            f"engine.{','.join(legacy_eng)} removed in P1.6 — LMCache is enabled by "
            f"providing engine.lmcache_config and setting workload.hit_source='lmcache'. "
            f"Move to a separate _lmcache.yaml overlay."
        )
    engine = EngineSpec(**engine_raw)

    workload_raw = dict(raw.get("workload", {}))
    legacy_wl = [k for k in _LEGACY_WORKLOAD_KEYS if k in workload_raw]
    if legacy_wl:
        raise ValueError(
            f"workload.{','.join(legacy_wl)} removed in P1.4 — length is the ctx_tokens "
            f"sweep axis; concurrency is the batch_sizes sweep axis."
        )
    workload = WorkloadSpec(**workload_raw)

    results_dir = Path(raw.get("results_dir", "results"))
    return SweepConfig(axes=axes, model=model, engine=engine,
                       workload=workload, results_dir=results_dir)
