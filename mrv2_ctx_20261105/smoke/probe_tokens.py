"""Send the fixed probe prompts to a running server at temperature 0 and save the
returned token ids. All prompts are in flight together.

  python probe_tokens.py <prompts.jsonl> <out.jsonl> [--port 8000] [--max-tokens 256]

prompts.jsonl rows: {"id": str, "prompt": str}
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def complete(port, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models") as r:
        model = json.load(r)["data"][0]["id"]
    with ThreadPoolExecutor(len(prompts)) as pool:
        results = list(
            pool.map(
                lambda p: complete(args.port, model, p["prompt"], args.max_tokens),
                prompts,
            )
        )
    with open(args.out, "w") as f:
        for p, res in zip(prompts, results):
            choice = res["choices"][0]
            f.write(
                json.dumps(
                    {
                        "id": p["id"],
                        "token_ids": choice.get("token_ids"),
                        "finish_reason": choice.get("finish_reason"),
                        "completion_tokens": res.get("usage", {}).get("completion_tokens"),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
