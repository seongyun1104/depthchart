from __future__ import annotations

import textwrap

import pytest

from benchmarks.sweep_config import _from_dict


def _yaml(text: str) -> dict:
    import yaml
    return yaml.safe_load(textwrap.dedent(text))


def test_v2_arm_axis_loads():
    raw = _yaml(
        """
        axes:
          batch_sizes: [128, 192, 256]
          ctx_tokens: [460, 970, 1990]
          spec_k: [0]
          spec_arms: [dsd_k0, no_spec]
          spec_arm_reference_k: 3
          spec_arm_batch_ceiling: 256
        model:
          name: gemma-4-31b-qat-fp8
          hf_id: prithivMLmods/gemma-4-31B-it-qat-FP8
          spec_methods: [mtp]
          max_context: 32768
          quantization: fp8
          draft_model: google/gemma-4-31B-it-qat-q4_0-unquantized-assistant
        workload:
          seeds: 3
        """
    )
    cfg = _from_dict(raw)
    assert cfg.axes.spec_arms == ("dsd_k0", "no_spec")
    assert cfg.axes.spec_arm_reference_k == 3
    assert cfg.axes.spec_arm_batch_ceiling == 256
    assert cfg.workload.seeds == 3


def test_spec_arms_and_nonzero_spec_k_rejected():
    raw = _yaml(
        """
        axes:
          batch_sizes: [256]
          ctx_tokens: [1990]
          spec_k: [0, 3]
          spec_arms: [dsd_k0, no_spec]
        model:
          name: gemma-4-31b-qat-fp8
          hf_id: prithivMLmods/gemma-4-31B-it-qat-FP8
          spec_methods: [mtp]
          max_context: 32768
          quantization: fp8
          draft_model: google/gemma-4-31B-it-qat-q4_0-unquantized-assistant
        """
    )
    with pytest.raises(ValueError, match="spec_arms"):
        _from_dict(raw)


def test_spec_arms_none_default_keeps_k_axis():
    raw = _yaml(
        """
        axes:
          batch_sizes: [32]
          ctx_tokens: [256, 2048]
          spec_k: [0, 3]
        model:
          name: gemma-4-31b-qat-fp8
          hf_id: prithivMLmods/gemma-4-31B-it-qat-FP8
          spec_methods: [mtp]
          max_context: 32768
          quantization: fp8
        """
    )
    cfg = _from_dict(raw)
    assert cfg.axes.spec_arms is None
    assert cfg.axes.spec_k == (0, 3)
