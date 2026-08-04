from __future__ import annotations

from pathlib import Path
from typing import IO

from dspark_trace_sim.trace_format import Provenance, StepRecord


class TraceLogger:
    def __init__(self, out_dir: Path, provenance: Provenance):
        self.out_dir = Path(out_dir)
        self.provenance = provenance
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._sample_id: str | None = None
        self._dataset: str | None = None
        self._step_idx: int = 0
        self._current: IO[str] | None = None

    def start_sample(self, sample_id: str, dataset: str) -> None:
        if self._current is not None:
            raise RuntimeError(
                f"Sample {self._sample_id!r} was not closed before starting "
                f"{sample_id!r}. Call end_sample() first."
            )
        dataset_dir = self.out_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = dataset_dir / f"{sample_id}.jsonl"
        self._current = path.open("w", encoding="utf-8")
        self._current.write(self.provenance.to_jsonl_line() + "\n")
        self._sample_id = sample_id
        self._dataset = dataset
        self._step_idx = 0

    def observe(
        self,
        confidences: list[float],
        accepts: list[int],
        prefix_len: int,
    ) -> None:
        if self._current is None or self._sample_id is None:
            raise RuntimeError(
                "No sample is currently open; call start_sample() first."
            )
        record = StepRecord(
            sample_id=self._sample_id,
            step_idx=self._step_idx,
            confidences=list(confidences),
            accepts=list(accepts),
            prefix_len=int(prefix_len),
        )
        self._current.write(record.to_jsonl_line() + "\n")
        self._step_idx += 1

    def end_sample(self) -> None:
        if self._current is None:
            return
        self._current.close()
        self._current = None
        self._sample_id = None
        self._dataset = None
        self._step_idx = 0

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, *_exc) -> None:
        self.end_sample()
