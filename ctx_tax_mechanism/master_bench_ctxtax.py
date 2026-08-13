#!/usr/bin/env python3
"""
ctx-tax mechanism master bench (pre-reg: ctx_tax_mechanism/PREREGISTRATION.md).

Sweeps the DSD baseline tax (no_spec vs dsd_k0) across ctx to 49.4k on the
Gemma-4-31B hybrid + MTP stack, V1 runner only, and captures scheduler /
prefix-cache / chunked-prefill instrumentation so the ctx-scaling can be
attributed (Suppressor72's invited #49986 diagnostic).

Adapted from tax_decomposition/master_bench_tax.py; the graph-mode V1/V2 lever
is dropped (already decomposed there), ctx grid + max-model-len + mechanism
logging are added.

NOT YET RUN. Concurrency per ctx and max_model_len KV headroom must be set on
the rental after measuring the KV pool (RUNBOOK §KV). DEBUG scheduler log format
must be confirmed against the running vLLM build before trusting the parser.

Usage:
  RESULTS_DIR=/root/results MAX_MODEL_LEN=52224 python master_bench_ctxtax.py no_spec dsd_k0
  # nsys wrap of arm B at the longest ctx (separate invocation):
  RESULTS_DIR=/root/results NSYS=1 python master_bench_ctxtax.py dsd_k0 --only-ctx 49400
"""
import argparse, json, os, shutil, signal, subprocess, time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

MODEL = os.environ.get("MODEL", "/root/models/target")
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "52224"))
NSYS = os.environ.get("NSYS", "0") == "1"
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "512"))

ARMS = {"no_spec": False, "dsd_k0": True}
DSD_K0_SCHEDULE = [[1, 64, 3], [65, 128, 0], [129, 512, 0]]

# Per-ctx concurrency: SET ON RENTAL from measured KV pool (RUNBOOK §KV).
# Defaults below are placeholders scaled ~1/ctx off the tax_decomposition anchor
# (55,215-token pool at 8192/0.90). Must keep arm B in the K=0 tier (batch>=129)
# where feasible; at high ctx the pool caps concurrency below 129 -> the tax is
# then read in the low-batch regime and that is recorded, not silently mixed.
CTX_CONCURRENCY = {
    400: 256,
    4000: 192,
    16000: 48,
    32000: 24,
    49400: 12,
}

# Metrics scraped as deltas around each measured run (superset of tax_decomp).
DELTA_METRICS = [
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
]


def wipe_autotune():
    p = Path.home() / ".cache/vllm/flashinfer_autotune_cache"
    shutil.rmtree(p, ignore_errors=True)


def spec_config(arm):
    if not ARMS[arm]:
        return None
    return {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "num_speculative_tokens_per_batch_size": DSD_K0_SCHEDULE,
    }


def launch_server(arm, tag):
    wipe_autotune()
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enable-prefix-caching",
    ]
    cfg = spec_config(arm)
    if cfg is not None:
        cmd += ["--speculative-config", json.dumps(cfg)]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"  # scheduler step census
    log_path = f"/tmp/server_{tag}.log"
    log = open(log_path, "w")
    if NSYS and ARMS[arm]:
        cmd = ["nsys", "profile", "-o", str(RESULTS_DIR / f"nsys_{tag}"),
               "-t", "cuda,nvtx", "--force-overwrite", "true", "--"] + cmd
    log.write(f"CMD: {' '.join(cmd)}\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    print(f"[{tag}] server PID={proc.pid}, waiting /health...", flush=True)
    t0 = time.time()
    timeout = 900
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            kill_server(proc, log)
            raise RuntimeError(f"Server {tag} exited early rc={proc.returncode} (see {log_path})")
        try:
            with urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[{tag}] READY after {time.time()-t0:.1f}s", flush=True)
                    return proc, log
        except (URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    kill_server(proc, log)
    raise RuntimeError(f"Server {tag} failed to start (see {log_path})")


def record_cudagraph_mode(tag):
    log = Path(f"/tmp/server_{tag}.log").read_text(errors="ignore")
    modes = [l for l in log.split("\n") if "cudagraph_mode" in l or "Dynamic speculative" in l]
    (RESULTS_DIR / f"cudagraph_{tag}.txt").write_text("\n".join(modes[-40:]))


def kill_server(proc, log):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    if log and not log.closed:
        log.close()
    time.sleep(8)


def get_metrics():
    try:
        with urlopen(f"{BASE_URL}/metrics", timeout=5) as r:
            return r.read().decode()
    except Exception:
        return ""


def parse_metric(text, name):
    total = 0.0
    for line in text.split("\n"):
        if line.startswith(name + " ") or (line.startswith(name) and "{" in line):
            try:
                total += float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    return total


def snapshot_delta(m0, m1, elapsed):
    d = {"elapsed_sec": elapsed}
    for m in DELTA_METRICS:
        d[m.replace("vllm:", "").replace("_total", "_delta")] = parse_metric(m1, m) - parse_metric(m0, m)
    return d


def run_bench(ctx, out_dir, out_file):
    out_dir.mkdir(parents=True, exist_ok=True)
    concurrency = CTX_CONCURRENCY[ctx]
    cmd = [
        "vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT),
        "--num-prompts", str(NUM_PROMPTS), "--max-concurrency", str(concurrency),
        "--ignore-eos", "--dataset-name", "prefix_repetition",
        "--prefix-repetition-prefix-len", str(ctx),
        "--prefix-repetition-suffix-len", "96",
        "--prefix-repetition-num-prefixes", "1",
        "--prefix-repetition-output-len", "100",
        "--percentile-metrics", "ttft,tpot,itl",
        "--save-result", "--result-dir", str(out_dir), "--result-filename", out_file,
    ]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        (out_dir / f"{out_file}.stderr").write_text(r.stderr[-4000:])
    return elapsed, r.returncode


def cold_start_burn():
    burn_dir = RESULTS_DIR / "burn"
    burn_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        cmd = [
            "vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT),
            "--num-prompts", "64", "--max-concurrency", "64", "--ignore-eos",
            "--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "64",
            "--save-result", "--result-dir", str(burn_dir), "--result-filename", f"burn_{i}.json",
        ]
        env = os.environ.copy()
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    time.sleep(30)


def phase_grid(arm, only_ctx):
    ctxs = [only_ctx] if only_ctx else list(CTX_CONCURRENCY)
    for ctx in ctxs:
        out_dir = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        reps = 4 if ctx >= 32000 else 6  # high-ctx runs are slow; 1 warmup + 3 measure
        n_warm = reps - 3
        for i in range(reps):
            is_measure = i >= n_warm
            label = "measure" if is_measure else "warmup"
            seed = i - n_warm if is_measure else i
            m0 = get_metrics()
            elapsed, rc = run_bench(ctx, out_dir, f"{label}_{seed}.json")
            m1 = get_metrics()
            if m0 and m1:
                (out_dir / f"snapshot_{label}_{seed}.json").write_text(
                    json.dumps(snapshot_delta(m0, m1, elapsed), indent=2))
            print(f"    [{arm}][ctx={ctx}][{label}_{seed}] elapsed={elapsed:.1f}s rc={rc}", flush=True)


def run_arm(arm, only_ctx):
    tag = f"{arm}{'_nsys' if NSYS else ''}{f'_ctx{only_ctx}' if only_ctx else ''}"
    proc, log = launch_server(arm, tag)
    try:
        record_cudagraph_mode(tag)
        cold_start_burn()
        phase_grid(arm, only_ctx)
    finally:
        # copy the DEBUG server log for the scheduler-census parser
        shutil.copy(f"/tmp/server_{tag}.log", RESULTS_DIR / f"server_{tag}.log")
        kill_server(proc, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", choices=list(ARMS))
    ap.add_argument("--only-ctx", type=int, default=0)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        run_arm(arm, args.only_ctx or None)


if __name__ == "__main__":
    main()
