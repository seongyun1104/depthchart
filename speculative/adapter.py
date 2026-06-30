from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpecMethod = Literal["off", "nextn", "fastmtp", "dflash", "eagle3"]


@dataclass(frozen=True)
class SpecConfig:
    method: SpecMethod
    num_speculative_tokens: int
    draft_model: str | None = None

    def to_sglang_args(self) -> list[str]:
        if self.method == "off" or self.num_speculative_tokens == 0:
            return []
        args = [
            "--speculative-algorithm", _to_sglang_algo(self.method),
            "--speculative-num-steps", str(self.num_speculative_tokens),
        ]
        if self.draft_model:
            args += ["--speculative-draft-model-path", self.draft_model]
        return args


def _to_sglang_algo(method: SpecMethod) -> str:
    # SGLang's flag spelling for spec algorithm. NEXTN covers DeepSeek-style
    # native MTP; EAGLE3 covers EXAONE 4.5's draft head; DFlash is Spec V2.
    return {
        "nextn": "NEXTN",
        "eagle3": "EAGLE3",
        "fastmtp": "EAGLE3",  # FastMTP head registered as EAGLE3-shape draft
        "dflash": "DFLASH",
    }[method]


def for_exaone_45(k: int) -> SpecConfig:
    return SpecConfig(method="eagle3" if k > 0 else "off",
                      num_speculative_tokens=k)
