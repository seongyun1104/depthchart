from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


SpecMethod = Literal["off", "nextn", "fastmtp", "dflash", "eagle3"]
Quantization = Literal["none", "fp8", "awq", "gptq"]


@dataclass(frozen=True)
class SweepAxes:
    batch_sizes: tuple[int, ...] = (1, 4, 16, 32, 64, 128)
    hit_rates: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
    spec_k: tuple[int, ...] = (0, 1, 2, 3)
    spec_method: SpecMethod = "eagle3"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    spec_methods: tuple[SpecMethod, ...]
    max_context: int
    quantization: Quantization = "none"


@dataclass(frozen=True)
class EngineSpec:
    host: str = "127.0.0.1"
    port: int = 30000
    enable_lmcache: bool = True
    chunked_prefill: bool = True
    tp_size: int = 1
    quantization_override: Quantization | None = None


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
            spec_methods=("eagle3",),
            max_context=32768,
            quantization="fp8",
        )
    )
    engine: EngineSpec = field(default_factory=EngineSpec)
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    results_dir: Path = field(default_factory=lambda: Path("results"))


def load(path: str | Path) -> SweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return _from_dict(raw)


def _from_dict(raw: dict) -> SweepConfig:
    axes = SweepAxes(**raw.get("axes", {}))
    model = ModelSpec(**raw["model"])
    engine = EngineSpec(**raw.get("engine", {}))
    workload = WorkloadSpec(**raw.get("workload", {}))
    results_dir = Path(raw.get("results_dir", "results"))
    return SweepConfig(axes=axes, model=model, engine=engine,
                       workload=workload, results_dir=results_dir)
