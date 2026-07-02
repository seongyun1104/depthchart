from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpecMethod = Literal["off", "mtp", "nextn", "eagle", "eagle3", "fastmtp", "dflash"]


@dataclass(frozen=True)
class SpecConfig:
    method: SpecMethod
    num_steps: int
    eagle_topk: int = 1
    num_draft_tokens: int = 4
    draft_model: str | None = None

    def to_sglang_args(self) -> list[str]:
        if self.method == "off" or self.num_steps == 0:
            return []
        args = [
            "--speculative-algorithm", _to_sglang_algo(self.method),
            "--speculative-num-steps", str(self.num_steps),
            "--speculative-eagle-topk", str(self.eagle_topk),
            "--speculative-num-draft-tokens", str(self.num_draft_tokens),
        ]
        if self.draft_model:
            args += ["--speculative-draft-model-path", self.draft_model]
        return args

    def to_vllm_speculative_config(self) -> dict | None:
        if self.method == "off" or self.num_steps == 0:
            return None
        cfg: dict = {
            "method": _to_vllm_method(self.method),
            "num_speculative_tokens": self.num_steps,
        }
        if self.draft_model:
            cfg["model"] = self.draft_model
        return cfg


def _to_sglang_algo(method: SpecMethod) -> str:
    return {
        "mtp": "MTP",
        "nextn": "NEXTN",
        "eagle": "EAGLE",
        "eagle3": "EAGLE3",
        "fastmtp": "EAGLE3",
        "dflash": "DFLASH",
    }[method]


def _to_vllm_method(method: SpecMethod) -> str:
    # vLLM routes MTP heads through the EAGLE proposer path; the "mtp" method
    # keyword selects that path for models that ship an EAGLE-style draft head
    # (EXAONE 4.5 native MTP head is one such case).
    return {
        "mtp": "mtp",
        "nextn": "mtp",
        "eagle": "eagle",
        "eagle3": "eagle3",
        "fastmtp": "eagle3",
    }[method]


def for_exaone_45(k: int, method: SpecMethod = "mtp") -> SpecConfig:
    if k <= 0:
        return SpecConfig(method="off", num_steps=0)
    return SpecConfig(
        method=method,
        num_steps=k,
        eagle_topk=1,
        num_draft_tokens=max(4, k + 1),
    )
