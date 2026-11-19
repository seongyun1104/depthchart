"""Write the smoke probe set: 8 prompts of ~4096 and 8 of ~32768 tokens under the
target tokenizer, from fixed text (no randomness).

  python make_probe_prompts.py <tokenizer dir> <out.jsonl>
"""

import json
import os
import sys

from transformers import AutoTokenizer

_lengths = [int(x) for x in os.environ.get("PROBE_LENGTHS", "4096,32768").split(",")]
LENGTHS = {"short": _lengths[0], "long": _lengths[1]}
PER_LENGTH = int(os.environ.get("PROBE_PER_LENGTH", "8"))


def text(seed, n_lines):
    return "\n".join(
        f"Record {seed}-{i}: item {(i * 7 + seed) % 97} moved from shelf "
        f"{(i * 13 + seed) % 31} to shelf {(i * 17 + seed) % 29} at step {i}."
        for i in range(n_lines)
    )


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        for name, target in LENGTHS.items():
            for j in range(PER_LENGTH):
                seed = j + (100 if name == "long" else 0)
                ids = tok(text(seed, target // 10 + 50))["input_ids"][: target - 32]
                body = tok.decode(ids, skip_special_tokens=True)
                prompt = body + "\nSummarize where each item ended up."
                f.write(
                    json.dumps(
                        {
                            "id": f"{name}_{j}",
                            "prompt": prompt,
                            "approx_tokens": len(tok(prompt)["input_ids"]),
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
