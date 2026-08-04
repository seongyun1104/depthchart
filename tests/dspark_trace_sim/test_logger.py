from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from dspark_trace_sim.deepspec_adapter import to_plain_arrays
from dspark_trace_sim.logger import TraceLogger
from dspark_trace_sim.trace_format import Provenance, read_trace

_PROVENANCE_KWARGS = {
    "deepspec_commit": "005e03b81cec38b7da6399833d609ee89a2587f2",
    "checkpoint_id": "deepseek-ai/dspark_gemma4_12b_block7",
    "checkpoint_revision": "abcdef1234567890",
    "target_model": "google/gemma-4-12B-it",
    "dataset": "gsm8k",
    "sampling_config": {"temperature": 1.0, "seed": 980406},
    "collected_at": "2026-08-04T12:34:56Z",
}


@dataclass
class _FakeProposal:
    confidence_logits: torch.Tensor | None


@dataclass
class _FakeVerification:
    accept_prefix_mask: torch.Tensor | None
    accepted_draft_tokens: int


def _fake_pair(
    logits: list[float],
    accepts: list[int],
    prefix_len: int,
) -> tuple[_FakeProposal, _FakeVerification]:
    proposal = _FakeProposal(confidence_logits=torch.tensor(logits))
    verification = _FakeVerification(
        accept_prefix_mask=torch.tensor(accepts, dtype=torch.int32),
        accepted_draft_tokens=prefix_len,
    )
    return proposal, verification


# ----- Adapter -----

def test_adapter_applies_sigmoid_to_confidences():
    logits = [-2.0, 0.0, 2.0]
    proposal, verification = _fake_pair(logits, [1, 0, 0], 1)

    confidences, accepts, prefix_len = to_plain_arrays(proposal, verification)

    expected = [1.0 / (1.0 + math.exp(-x)) for x in logits]
    assert len(confidences) == len(expected)
    for got, want in zip(confidences, expected, strict=True):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6)
    assert accepts == [1, 0, 0]
    assert prefix_len == 1


def test_adapter_flattens_multidimensional_tensors():
    logits_2d = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    proposal = _FakeProposal(confidence_logits=logits_2d)
    verification = _FakeVerification(
        accept_prefix_mask=torch.tensor([[1, 1], [0, 0]], dtype=torch.int32),
        accepted_draft_tokens=2,
    )

    confidences, accepts, prefix_len = to_plain_arrays(proposal, verification)

    assert len(confidences) == 4
    assert accepts == [1, 1, 0, 0]
    assert prefix_len == 2


def test_adapter_rejects_missing_confidence_logits():
    proposal = _FakeProposal(confidence_logits=None)
    verification = _FakeVerification(
        accept_prefix_mask=torch.tensor([1, 0]),
        accepted_draft_tokens=1,
    )
    with pytest.raises(ValueError, match="confidence_logits"):
        to_plain_arrays(proposal, verification)


def test_adapter_rejects_missing_accept_prefix_mask():
    proposal = _FakeProposal(confidence_logits=torch.tensor([0.0, 1.0]))
    verification = _FakeVerification(
        accept_prefix_mask=None,
        accepted_draft_tokens=0,
    )
    with pytest.raises(ValueError, match="accept_prefix_mask"):
        to_plain_arrays(proposal, verification)


# ----- Logger + end-to-end -----

def test_logger_writes_provenance_and_records(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    logger = TraceLogger(tmp_path, provenance)

    proposal, verification = _fake_pair([2.0, 1.0, -1.0, -2.0], [1, 1, 0, 0], 2)
    confidences, accepts, prefix_len = to_plain_arrays(proposal, verification)

    logger.start_sample("gsm8k_001", "gsm8k")
    logger.observe(confidences, accepts, prefix_len)
    logger.observe(confidences, accepts, prefix_len)
    logger.end_sample()

    trace_path = tmp_path / "gsm8k" / "gsm8k_001.jsonl"
    read_provenance, records = read_trace(trace_path)
    assert read_provenance == provenance
    assert len(records) == 2
    assert [r.step_idx for r in records] == [0, 1]
    assert records[0].sample_id == "gsm8k_001"


def test_logger_forbids_observe_without_start(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    logger = TraceLogger(tmp_path, provenance)

    with pytest.raises(RuntimeError, match="No sample"):
        logger.observe([0.5], [1], 1)


def test_logger_forbids_double_start(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    logger = TraceLogger(tmp_path, provenance)

    logger.start_sample("a", "gsm8k")
    with pytest.raises(RuntimeError, match="not closed"):
        logger.start_sample("b", "gsm8k")
    logger.end_sample()


def test_logger_context_manager_closes_open_sample(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    with TraceLogger(tmp_path, provenance) as logger:
        logger.start_sample("gsm8k_ctx", "gsm8k")
        logger.observe([0.9, 0.1], [1, 0], 1)

    trace_path = tmp_path / "gsm8k" / "gsm8k_ctx.jsonl"
    read_provenance, records = read_trace(trace_path)
    assert read_provenance == provenance
    assert len(records) == 1


def test_logger_supports_multiple_datasets(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    logger = TraceLogger(tmp_path, provenance)

    logger.start_sample("s1", "gsm8k")
    logger.observe([0.9, 0.5], [1, 0], 1)
    logger.end_sample()

    logger.start_sample("s1", "mt-bench")
    logger.observe([0.4, 0.3], [0, 0], 0)
    logger.end_sample()

    assert (tmp_path / "gsm8k" / "s1.jsonl").exists()
    assert (tmp_path / "mt-bench" / "s1.jsonl").exists()
