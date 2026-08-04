from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from dspark_trace_sim.trace_format import (
    TRACE_SCHEMA_VERSION,
    Provenance,
    StepRecord,
    read_trace,
    write_trace,
)


_PROVENANCE_KWARGS = {
    "deepspec_commit": "005e03b81cec38b7da6399833d609ee89a2587f2",
    "checkpoint_id": "deepseek-ai/dspark_gemma4_12b_block7",
    "checkpoint_revision": "abcdef1234567890",
    "target_model": "google/gemma-4-12B-it",
    "dataset": "gsm8k",
    "sampling_config": {"temperature": 1.0, "seed": 980406},
    "collected_at": "2026-08-04T12:34:56Z",
}


def _step(**overrides):
    base = {
        "sample_id": "gsm8k_042",
        "step_idx": 0,
        "confidences": [0.87, 0.62, 0.41, 0.29, 0.18, 0.11, 0.07],
        "accepts": [1, 1, 0, 0, 0, 0, 0],
        "prefix_len": 2,
    }
    base.update(overrides)
    return StepRecord(**base)


# ----- Roundtrip -----

def test_roundtrip_write_and_read(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    records = [_step(step_idx=i) for i in range(3)]
    path = tmp_path / "gsm8k" / "sample_042.jsonl"

    write_trace(path, provenance, records)
    read_provenance, read_records = read_trace(path)

    assert read_provenance == provenance
    assert read_records == records


def test_provenance_jsonl_roundtrip_preserves_sentinel(tmp_path):
    provenance = Provenance(**_PROVENANCE_KWARGS)
    line = provenance.to_jsonl_line()
    payload = json.loads(line)

    assert payload["__provenance__"] is True
    assert Provenance.from_jsonl_line(line) == provenance


def test_missing_sentinel_rejected():
    payload = dict(**_PROVENANCE_KWARGS, trace_schema_version=TRACE_SCHEMA_VERSION)
    line = json.dumps(payload)

    with pytest.raises(ValueError, match="__provenance__"):
        Provenance.from_jsonl_line(line)


# ----- Provenance mandatory-field enforcement -----

@pytest.mark.parametrize(
    "field",
    [
        "deepspec_commit",
        "checkpoint_id",
        "checkpoint_revision",
        "target_model",
        "dataset",
        "collected_at",
    ],
)
def test_provenance_rejects_missing_mandatory_field(field):
    kwargs = copy.deepcopy(_PROVENANCE_KWARGS)
    del kwargs[field]

    with pytest.raises(ValidationError):
        Provenance(**kwargs)


@pytest.mark.parametrize(
    "field, empty_value",
    [
        ("deepspec_commit", ""),
        ("checkpoint_id", ""),
        ("checkpoint_revision", ""),
        ("target_model", ""),
        ("dataset", ""),
        ("collected_at", ""),
    ],
)
def test_provenance_rejects_empty_mandatory_field(field, empty_value):
    kwargs = copy.deepcopy(_PROVENANCE_KWARGS)
    kwargs[field] = empty_value

    with pytest.raises(ValidationError):
        Provenance(**kwargs)


def test_provenance_rejects_short_commit_sha():
    kwargs = copy.deepcopy(_PROVENANCE_KWARGS)
    kwargs["deepspec_commit"] = "abc"

    with pytest.raises(ValidationError):
        Provenance(**kwargs)


def test_provenance_rejects_unknown_field():
    kwargs = copy.deepcopy(_PROVENANCE_KWARGS)
    kwargs["extra_field"] = "unexpected"

    with pytest.raises(ValidationError):
        Provenance(**kwargs)


def test_provenance_defaults_schema_version():
    provenance = Provenance(**_PROVENANCE_KWARGS)
    assert provenance.trace_schema_version == TRACE_SCHEMA_VERSION


# ----- StepRecord invariants -----

def test_confidences_out_of_unit_interval_rejected():
    with pytest.raises(ValidationError, match=r"post-sigmoid"):
        _step(confidences=[0.5, 1.5, 0.2, 0.1, 0.05, 0.01, 0.001])


def test_negative_confidence_rejected():
    with pytest.raises(ValidationError, match=r"post-sigmoid"):
        _step(confidences=[-0.1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])


def test_accepts_non_binary_rejected():
    with pytest.raises(ValidationError, match=r"not in"):
        _step(accepts=[1, 2, 0, 0, 0, 0, 0])


def test_length_mismatch_rejected():
    with pytest.raises(ValidationError, match=r"length"):
        _step(
            confidences=[0.9, 0.8, 0.7],
            accepts=[1, 0],
            prefix_len=1,
        )


def test_prefix_len_disagreement_rejected():
    with pytest.raises(ValidationError, match=r"leading-1"):
        _step(
            confidences=[0.9, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01],
            accepts=[1, 1, 0, 0, 0, 0, 0],
            prefix_len=3,
        )


def test_prefix_len_zero_when_first_reject():
    record = _step(
        confidences=[0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001],
        accepts=[0, 0, 0, 0, 0, 0, 0],
        prefix_len=0,
    )
    assert record.prefix_len == 0


def test_prefix_len_full_when_all_accepted():
    record = _step(
        confidences=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        accepts=[1, 1, 1, 1, 1, 1, 1],
        prefix_len=7,
    )
    assert record.prefix_len == 7


def test_step_record_jsonl_roundtrip():
    record = _step()
    line = record.to_jsonl_line()
    assert StepRecord.from_jsonl_line(line) == record


def test_empty_trace_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="Empty trace file"):
        read_trace(path)
