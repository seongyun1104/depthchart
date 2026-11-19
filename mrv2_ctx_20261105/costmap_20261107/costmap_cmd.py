"""argv for the server and the bench client; shared by the runner and the CLI check."""

import json

from policy_nd import nd_spec as ND

OUT_LEN = 640
SERVE_FLAGS = ["--attention-backend", "TRITON_ATTN", "--no-enable-prefix-caching",
               "--max-num-seqs", "64", "--max-model-len", "34816", "--gpu-memory-utilization", "0.90"]


def spec(config, draft):
    if config == "OFF":
        return None
    if config == "Z":
        return {"model": draft, "num_speculative_tokens": 1,
                "num_speculative_tokens_per_batch_size": [[1, 64, 0]]}
    if config in ND.TABLES:
        return {"model": draft, "num_speculative_tokens": ND.MAX_K,
                "num_speculative_tokens_per_batch_size_and_context": ND.TABLES[config]}
    return {"model": draft, "num_speculative_tokens": int(config[1:])}


def serve_argv(model, s, port):
    return ["vllm", "serve", model, "--port", str(port), *SERVE_FLAGS] + \
        (["--speculative-config", json.dumps(s)] if s else [])


def bench_argv_natural(model, port, cell, result_dir, dataset_path, out_len, tokenizer=None):
    """Closed batch of the cell's prompts (pre-templated text, file order), natural EOS."""
    return ["vllm", "bench", "serve", "--model", model, "--port", str(port),
            *(["--tokenizer", tokenizer] if tokenizer else []),
            "--dataset-name", "custom", "--dataset-path", str(dataset_path), "--disable-shuffle",
            "--skip-chat-template", "--custom-output-len", str(out_len),
            "--num-prompts", str(cell["bs"]), "--max-concurrency", str(cell["bs"]),
            "--temperature", "0", "--seed", str(cell["seed"]),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--save-result", "--save-detailed",
            "--result-dir", str(result_dir), "--result-filename", f"{cell['id']}.json"]


def bench_argv(model, port, cell, result_dir, tokenizer=None):
    return ["vllm", "bench", "serve", "--model", model, "--port", str(port),
            *(["--tokenizer", tokenizer] if tokenizer else []),
            "--dataset-name", "random", "--random-input-len", str(cell["ctx"]),
            "--random-output-len", str(OUT_LEN), "--random-range-ratio", "0",
            "--num-prompts", str(cell["bs"]), "--max-concurrency", str(cell["bs"]),
            "--ignore-eos", "--temperature", "0", "--seed", str(cell["seed"]),
            "--percentile-metrics", "ttft,tpot,itl",
            "--save-result", "--result-dir", str(result_dir), "--result-filename", f"{cell['id']}.json"]
