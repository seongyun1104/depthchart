"""argv for the server and the bench client; shared by the runner and the CLI check."""

import json

OUT_LEN = 640
SERVE_FLAGS = ["--attention-backend", "TRITON_ATTN", "--no-enable-prefix-caching",
               "--max-num-seqs", "64", "--max-model-len", "34816", "--gpu-memory-utilization", "0.90"]


def spec(config, draft):
    if config == "OFF":
        return None
    if config == "Z":
        return {"model": draft, "num_speculative_tokens": 1,
                "num_speculative_tokens_per_batch_size": [[1, 64, 0]]}
    return {"model": draft, "num_speculative_tokens": int(config[1:])}


def serve_argv(model, s, port):
    return ["vllm", "serve", model, "--port", str(port), *SERVE_FLAGS] + \
        (["--speculative-config", json.dumps(s)] if s else [])


def bench_argv(model, port, cell, result_dir, tokenizer=None):
    return ["vllm", "bench", "serve", "--model", model, "--port", str(port),
            *(["--tokenizer", tokenizer] if tokenizer else []),
            "--dataset-name", "random", "--random-input-len", str(cell["ctx"]),
            "--random-output-len", str(OUT_LEN), "--random-range-ratio", "0",
            "--num-prompts", str(cell["bs"]), "--max-concurrency", str(cell["bs"]),
            "--ignore-eos", "--temperature", "0", "--seed", str(cell["seed"]),
            "--percentile-metrics", "ttft,tpot,itl",
            "--save-result", "--result-dir", str(result_dir), "--result-filename", f"{cell['id']}.json"]
