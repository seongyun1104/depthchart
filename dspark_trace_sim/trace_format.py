from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TRACE_SCHEMA_VERSION = "1.0"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_schema_version: str = Field(default=TRACE_SCHEMA_VERSION, min_length=1)
    deepspec_commit: str = Field(..., min_length=7)
    checkpoint_id: str = Field(..., min_length=1)
    checkpoint_revision: str = Field(..., min_length=7)
    target_model: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    sampling_config: dict = Field(default_factory=dict)
    collected_at: str = Field(..., min_length=1)

    def to_jsonl_line(self) -> str:
        payload = {"__provenance__": True, **self.model_dump()}
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_jsonl_line(cls, line: str) -> Provenance:
        data = json.loads(line)
        if not data.pop("__provenance__", False):
            raise ValueError(
                "Missing __provenance__ sentinel; first line of a trace file must "
                "be a provenance header."
            )
        return cls.model_validate(data)


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)
    step_idx: int = Field(..., ge=0)
    confidences: list[float] = Field(..., min_length=1)
    accepts: list[int] = Field(..., min_length=1)
    prefix_len: int = Field(..., ge=0)

    @field_validator("confidences")
    @classmethod
    def _probs_in_unit_interval(cls, v: list[float]) -> list[float]:
        for i, p in enumerate(v):
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"confidences[{i}]={p} out of [0, 1]. "
                    "Store post-sigmoid probabilities, not raw logits."
                )
        return v

    @field_validator("accepts")
    @classmethod
    def _accepts_binary(cls, v: list[int]) -> list[int]:
        for i, a in enumerate(v):
            if a not in (0, 1):
                raise ValueError(f"accepts[{i}]={a} not in {{0, 1}}.")
        return v

    @model_validator(mode="after")
    def _check_invariants(self) -> StepRecord:
        if len(self.confidences) != len(self.accepts):
            raise ValueError(
                f"confidences length {len(self.confidences)} != "
                f"accepts length {len(self.accepts)}"
            )
        leading_ones = 0
        for a in self.accepts:
            if a == 1:
                leading_ones += 1
            else:
                break
        if self.prefix_len != leading_ones:
            raise ValueError(
                f"prefix_len={self.prefix_len} disagrees with accepts leading-1 "
                f"count {leading_ones}; duplicate fields must be consistent."
            )
        return self

    def to_jsonl_line(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)

    @classmethod
    def from_jsonl_line(cls, line: str) -> StepRecord:
        return cls.model_validate_json(line)


@dataclass
class SampleTrace:
    sample_id: str
    dataset: str
    steps: list[StepRecord] = field(default_factory=list)


def write_trace(
    path: Path,
    provenance: Provenance,
    records: Iterable[StepRecord],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(provenance.to_jsonl_line() + "\n")
        for rec in records:
            f.write(rec.to_jsonl_line() + "\n")


def read_trace(path: Path) -> tuple[Provenance, list[StepRecord]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty trace file: {path}")
    provenance = Provenance.from_jsonl_line(lines[0])
    records = [StepRecord.from_jsonl_line(line) for line in lines[1:]]
    return provenance, records
