# Hardware-Aware Prefix Scheduler — paper §3.2.2 Algorithm 1 (arXiv:2607.05147).
#
# Two variants:
# - algorithm_1(..., unconstrained=False): paper §3.2.2, greedy admission with
#   early-stop `break` at the first throughput decrease.
# - algorithm_1(..., unconstrained=True): paper §5.2 asynchronous adaptation
#   ("we remove the early-stopping break, enabling an unconstrained global
#    search"). Retains rank-preserving admission.
#
# Third helper matches the PR admission semantics (fixed budget top-K):
# - admit_top_k: mirrors _assign_draft_token_budget in neuralmagic/vllm
#   @codex/dspark-capacity-realloc adaptive_verification.py L30-58. The PR
#   splits budget selection (get_num_tokens via stale EMA) from admission
#   (this function, with fresh confidences). See dspark_pr47808_analysis.md
#   in the private notes.

from __future__ import annotations

from collections.abc import Callable, Sequence


def _survival_candidates(
    confidences: Sequence[Sequence[float]],
) -> list[tuple[float, int, int]]:
    """Enumerate (survival_prob, request_idx, position_idx), skipping zeros.

    Returned list is sorted descending by survival, with (request_idx, position_idx)
    as stable tie-breakers so ordering is deterministic across runs.
    """
    survivals: list[tuple[float, int, int]] = []
    for r, conf_r in enumerate(confidences):
        cum = 1.0
        for j, c in enumerate(conf_r):
            cum *= c
            if cum > 0.0:
                survivals.append((cum, r, j))
    survivals.sort(key=lambda x: (-x[0], x[1], x[2]))
    return survivals


def algorithm_1(
    confidences: Sequence[Sequence[float]],
    sps: Callable[[int], float],
    *,
    unconstrained: bool = False,
) -> tuple[list[int], float]:
    """Paper §3.2.2 Algorithm 1.

    Args:
        confidences: [R][γ] post-sigmoid probabilities. Each row = one request.
        sps: profiled step curve, batch size B -> steps per second (higher is
             faster). Paper assumes non-increasing in B, but the function does
             not enforce this.
        unconstrained: False = paper §3.2.2 (early-stop break). True = §5.2
             (global search, keeps scanning after a throughput drop).

    Returns:
        (per-request admitted lengths, best throughput achieved).
    """
    num_requests = len(confidences)
    if num_requests == 0:
        return [], 0.0

    survivals = _survival_candidates(confidences)

    lengths = [0] * num_requests
    batch_size = num_requests
    tau_star = float(num_requests)
    theta_best = float(num_requests) * sps(num_requests)
    best_lengths = lengths.copy()

    for cum, r, j in survivals:
        lengths[r] = j + 1
        batch_size += 1
        tau_star += cum
        theta = tau_star * sps(batch_size)
        if theta > theta_best:
            theta_best = theta
            best_lengths = lengths.copy()
        elif not unconstrained:
            break

    return best_lengths, theta_best


def admit_top_k(
    confidences: Sequence[Sequence[float]],
    budget: int,
) -> list[int]:
    """Greedy top-K admission given a pre-determined draft budget.

    Mirrors PR's `_assign_draft_token_budget`: admits the `budget` slots with
    the highest survival probability across all (request, position) pairs.
    Contiguity within a request is preserved because survival is monotone
    non-increasing along positions, so (r, j-1) always precedes (r, j) in the
    sorted order.

    Args:
        confidences: [R][γ] post-sigmoid probabilities.
        budget: total number of draft slots to admit across the batch.

    Returns:
        Per-request admitted lengths summing to min(budget, valid slot count).
    """
    num_requests = len(confidences)
    if num_requests == 0:
        return []
    lengths = [0] * num_requests
    if budget <= 0:
        return lengths

    survivals = _survival_candidates(confidences)
    for admitted, (_cum, r, j) in enumerate(survivals):
        if admitted >= budget:
            break
        lengths[r] = j + 1
    return lengths
