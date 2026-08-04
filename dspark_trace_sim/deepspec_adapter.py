# Interface pinned to deepseek-ai/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2
# (deepspec/eval/dspark/evaluator.py::Qwen3DSparkEvaluator._post_verify,
#  deepspec/eval/dspark/draft_ops.py::DSparkDraftProposal.confidence_logits,
#  deepspec/eval/base_evaluator.py::VerificationResult.accept_prefix_mask).
#
# RE-VERIFY at Phase 2A collection time: if DeepSpec HEAD has drifted and any of
# the field names above changed, update this file only. logger.py and the rest of
# the pipeline consume plain arrays and are shielded from DeepSpec renames.

from __future__ import annotations

from typing import Protocol

import torch


class _Proposal(Protocol):
    confidence_logits: torch.Tensor | None


class _Verification(Protocol):
    accept_prefix_mask: torch.Tensor | None
    accepted_draft_tokens: int


def to_plain_arrays(
    proposal: _Proposal,
    verification: _Verification,
) -> tuple[list[float], list[int], int]:
    if proposal.confidence_logits is None:
        raise ValueError(
            "proposal.confidence_logits is None; DSpark confidence head absent "
            "or not populated for this step."
        )
    if verification.accept_prefix_mask is None:
        raise ValueError(
            "verification.accept_prefix_mask is None; verification did not run."
        )

    confidences = torch.sigmoid(proposal.confidence_logits).flatten().tolist()
    accepts = [int(x) for x in verification.accept_prefix_mask.flatten().tolist()]
    prefix_len = int(verification.accepted_draft_tokens)
    return confidences, accepts, prefix_len
