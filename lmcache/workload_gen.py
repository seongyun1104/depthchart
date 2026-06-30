from __future__ import annotations

import random
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Prompt:
    text: str
    prompt_tokens: int
    shared_prefix_tokens: int
    is_warm: bool


@dataclass
class HitRateController:
    """Generates prompts whose shared-prefix structure targets a specific
    LMCache hit rate. A prompt counts as 'warm' if it reuses a prefix that
    has already been served at least once in this run.
    """
    target_hit_rate: float
    prompt_tokens: int
    shared_prefix_tokens: int
    tokenizer: PreTrainedTokenizerBase
    seed: int = 0

    def __post_init__(self) -> None:
        assert 0.0 <= self.target_hit_rate <= 1.0
        assert 0 < self.shared_prefix_tokens < self.prompt_tokens
        self._rng = random.Random(self.seed)
        self._prefix_bank: list[list[int]] = []
        self._issued = 0
        self._warm_target = 0

    def _new_prefix(self) -> list[int]:
        ids = [self._rng.randint(1000, 30000) for _ in range(self.shared_prefix_tokens)]
        self._prefix_bank.append(ids)
        return ids

    def _pick_prefix(self) -> tuple[list[int], bool]:
        if not self._prefix_bank:
            return self._new_prefix(), False
        self._issued += 1
        self._warm_target = round(self._issued * self.target_hit_rate)
        warm_so_far = sum(1 for _ in range(self._issued - 1))  # placeholder counter
        # honor target rate roughly: if behind quota, reuse; else new
        if warm_so_far < self._warm_target:
            return self._rng.choice(self._prefix_bank), True
        return self._new_prefix(), False

    def next(self) -> Prompt:
        prefix_ids, is_warm = self._pick_prefix()
        suffix_len = self.prompt_tokens - self.shared_prefix_tokens
        suffix_ids = [self._rng.randint(1000, 30000) for _ in range(suffix_len)]
        ids = prefix_ids + suffix_ids
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return Prompt(
            text=text,
            prompt_tokens=self.prompt_tokens,
            shared_prefix_tokens=self.shared_prefix_tokens,
            is_warm=is_warm,
        )
