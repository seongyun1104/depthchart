from __future__ import annotations

import random
from dataclasses import dataclass, field

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
    _rng: random.Random = field(init=False)
    _prefix_bank: list[list[int]] = field(default_factory=list, init=False)
    _issued: int = field(default=0, init=False)
    _warm_issued: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        assert 0.0 <= self.target_hit_rate <= 1.0
        assert 0 < self.shared_prefix_tokens < self.prompt_tokens
        self._rng = random.Random(self.seed)

    def _new_prefix(self) -> list[int]:
        ids = [self._rng.randint(1000, 30000) for _ in range(self.shared_prefix_tokens)]
        self._prefix_bank.append(ids)
        return ids

    def _pick_prefix(self) -> tuple[list[int], bool]:
        self._issued += 1
        if not self._prefix_bank:
            self._new_prefix()
            return self._prefix_bank[-1], False
        target_warm = round(self._issued * self.target_hit_rate)
        if self._warm_issued < target_warm:
            self._warm_issued += 1
            return self._rng.choice(self._prefix_bank), True
        self._new_prefix()
        return self._prefix_bank[-1], False

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

    @property
    def realized_hit_rate(self) -> float:
        return self._warm_issued / self._issued if self._issued else 0.0
