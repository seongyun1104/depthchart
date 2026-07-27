#!/usr/bin/env python3
"""
Master bench script for PR #48944 verification.
Phase 4a: PR schedule (A_prime, C_prime) with prefix_repetition
Phase 4b: §2 reproduction (no_spec, static_k3) with prefix_repetition
Phase 1:  motivating cell (A_prime, C_prime, C_pp) with speed_bench/random
Phase 5:  spec_bench per-category (C_prime)
"""
import subprocess, json, time, sys, os, signal, shutil
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

MODEL = "/root/models/target"
DRAFT = "/root/models/draft"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))

SCHEDULES = {
    "A_prime": [[1, 64, 3], [65, 128, 1], [129, 512, 0]],  # 3-item batch-only
    "C_prime": [  # 6-cell 5-item, mid 장문 K=1 유지 (단일 셀 격리)
        [1, 64, 1, 768, 3], [1, 64, 769, 32768, 3],
        [65, 128, 1, 768, 1], [65, 128, 769, 32768, 1],
        [129, 512, 1, 768, 0], [129, 512, 769, 32768, 3],
    ],
    "C_pp": [  # 6-cell 5-item, mid 장문 K=3 (2-cell 안)
        [1, 64, 1, 768, 3], [1, 64, 769, 32768, 3],
        [65, 128, 1, 768, 1], [65, 128, 769, 32768, 3],
        [129, 512, 1, 768, 0], [129, 512, 769, 32768, 3],
    ],
    "static_k3": [[1, 512, 3]],  # §2 K=3 static
    # Eager control arms: §2 condition-matched (enforce_eager=True, max_model_len=4096)
    # to isolate cudagraph mode asymmetry as the confound (FULL_AND_PIECEWISE vs PIECEWISE)
    "eager_no_spec": None,  # no speculative_config
    "eager_static_k3": [[1, 512, 3]],
    # Recheck: C_prime ctx=400 at c=192 to eliminate preempt contamination
    "C_prime_recheck_c192": [  # same schedule as C_prime
        [1, 64, 1, 768, 3], [1, 64, 769, 32768, 3],
        [65, 128, 1, 768, 1], [65, 128, 769, 32768, 1],
        [129, 512, 1, 768, 0], [129, 512, 769, 32768, 3],
    ],
}


def wipe_autotune():
    shutil.rmtree(Path.home() / ".cache/vllm/flashinfer_autotune_cache", ignore_errors=True)


def spec_config(arm):
    if arm in ("no_spec", "eager_no_spec"):
        return None
    cfg = {
        "model": DRAFT,
        "num_speculative_tokens": 4,
        "num_speculative_tokens_per_batch_size": SCHEDULES[arm],
    }
    if arm in ("C_prime", "C_pp", "C_prime_recheck_c192"):
        cfg["ctx_agg"] = "mean"
    return cfg


def arm_options(arm):
    """Per-arm launch options. Eager control arms match §2 conditions."""
    if arm.startswith("eager_"):
        return {"enforce_eager": True, "max_model_len": 4096}
    return {"enforce_eager": False, "max_model_len": 8192}


def launch_server(arm):
    wipe_autotune()
    opts = arm_options(arm)
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(opts["max_model_len"]),
    ]
    if opts["enforce_eager"]:
        cmd.append("--enforce-eager")
    cfg = spec_config(arm)
    if cfg is not None:
        cmd += ["--speculative-config", json.dumps(cfg)]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONHASHSEED"] = "0"
    log_path = f"/tmp/server_{arm}.log"
    log = open(log_path, "w")
    log.write(f"CMD: {' '.join(cmd)}\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    print(f"[{arm}] server PID={proc.pid}, waiting for /health (timeout=600s)...", flush=True)
    t0 = time.time()
    last_status_print = 0
    while time.time() - t0 < 600:
        elapsed = time.time() - t0
        try:
            with urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[{arm}] server READY after {elapsed:.1f}s", flush=True)
                    return proc, log
        except (URLError, ConnectionError, OSError):
            pass
        if elapsed - last_status_print >= 30:
            print(f"    [{arm}] still waiting... elapsed={elapsed:.0f}s", flush=True)
            last_status_print = elapsed
        time.sleep(5)
    kill_server(proc, log)
    raise RuntimeError(f"Server {arm} failed to start in 600s (see {log_path})")


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
    print(f"    server killed, waiting KV free...", flush=True)
    time.sleep(8)


def get_metrics():
    try:
        with urlopen(f"{BASE_URL}/metrics", timeout=5) as r:
            return r.read().decode()
    except Exception as e:
        return ""


def parse_metric(text, name):
    total = 0.0
    for line in text.split("\n"):
        if line.startswith(name) and "{" in line and "}" in line:
            try:
                total += float(line.split()[-1])
            except (ValueError, IndexError):
                pass
        elif line.startswith(name + " "):
            try:
                total += float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    return total


def snapshot_delta(m_before, m_after, elapsed):
    metrics = [
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
    ]
    d = {"elapsed_sec": elapsed}
    for m in metrics:
        d[m.replace("vllm:", "").replace("_total", "_delta")] = (
            parse_metric(m_after, m) - parse_metric(m_before, m)
        )
    return d


def run_bench(dataset_args, out_dir, out_file, extra_env=None, concurrency=256, num_prompts=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if num_prompts is None:
        num_prompts = concurrency
    cmd = [
        "vllm", "bench", "serve",
        "--model", MODEL,
        "--port", str(PORT),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl",
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", out_file,
    ] + dataset_args
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0
    if r.returncode != 0:
        (out_dir / f"{out_file}.stderr").write_text(r.stderr[-4000:])
    return elapsed, r.returncode


def phase4_bench(arm):
    """Phase 4: prefix_repetition × ctx sweep, 3 warm + 3 measure.
    Eager control arms: crossover cells only (ctx 1900, 4000).
    C_prime recheck at c=192: ctx 400 only, isolates preempt contamination.
    """
    if arm == "C_prime_recheck_c192":
        ctxs = [400]
        concurrency = 192
    elif arm.startswith("eager_"):
        ctxs = [1900, 4000]
        concurrency = 256
    else:
        ctxs = [400, 900, 1900, 4000]
        concurrency = 256
    for ctx in ctxs:
        out_dir = RESULTS_DIR / "phase4" / arm / f"ctx_{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            is_measure = i >= 3
            label = "measure" if is_measure else "warmup"
            seed = i - 3 if is_measure else i
            fname = f"{label}_{seed}.json"
            ds = [
                "--dataset-name", "prefix_repetition",
                "--prefix-repetition-prefix-len", str(ctx),
                "--prefix-repetition-suffix-len", "96",
                "--prefix-repetition-num-prefixes", "1",
                "--prefix-repetition-output-len", "100",
            ]
            m0 = get_metrics()
            elapsed, rc = run_bench(ds, out_dir, fname, concurrency=concurrency)
            m1 = get_metrics()
            if m0 and m1:
                snap = snapshot_delta(m0, m1, elapsed)
                (out_dir / f"snapshot_{label}_{seed}.json").write_text(json.dumps(snap, indent=2))
            print(f"    [{arm}][ctx={ctx}][{label}_{seed}] elapsed={elapsed:.1f}s rc={rc}",
                  flush=True)


def phase1_bench(arm, dataset="random"):
    """Phase 1: motivating cell, c=192, speed_bench or random."""
    out_dir = RESULTS_DIR / "phase1" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        is_measure = i >= 3
        label = "measure" if is_measure else "warmup"
        seed = i - 3 if is_measure else i
        fname = f"{label}_{seed}.json"
        if dataset == "speed_bench":
            ds = ["--dataset-name", "speed_bench",
                  "--dataset-path", "/root/data/speed_bench",
                  "--speed-bench-dataset-subset", "throughput_2k"]
        else:
            ds = ["--dataset-name", "random",
                  "--random-input-len", "2000",
                  "--random-output-len", "200"]
        m0 = get_metrics() if is_measure else ""
        cmd_extra = ["--max-concurrency", "192"]  # override default 256
        # Rebuild args with c=192
        ds = ds + ["--max-concurrency", "192"] if False else ds
        # For simplicity use run_bench and manually override in dataset_args
        elapsed, rc = _run_bench_p1(ds, out_dir, fname, concurrency=192)
        m1 = get_metrics() if is_measure else ""
        if is_measure and m0 and m1:
            snap = snapshot_delta(m0, m1, elapsed)
            (out_dir / f"snapshot_{seed}.json").write_text(json.dumps(snap, indent=2))
        print(f"    [{arm}][phase1][{label}_{seed}] elapsed={elapsed:.1f}s rc={rc}", flush=True)


def _run_bench_p1(dataset_args, out_dir, out_file, concurrency=256):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm", "bench", "serve",
        "--model", MODEL,
        "--port", str(PORT),
        "--num-prompts", "128",
        "--max-concurrency", str(concurrency),
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl",
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", out_file,
    ] + dataset_args
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0
    if r.returncode != 0:
        (out_dir / f"{out_file}.stderr").write_text(r.stderr[-4000:])
    return elapsed, r.returncode


def phase5_bench(arm):
    """Phase 5: spec_bench per-category (13 categories), C_prime only."""
    categories = ["writing", "roleplay", "reasoning", "math", "coding", "extraction",
                  "stem", "humanities", "translation", "summarization", "qa",
                  "math_reasoning", "rag"]
    for cat in categories:
        out_dir = RESULTS_DIR / "phase5" / arm / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = ["--dataset-name", "spec_bench",
              "--dataset-path", "/root/data/spec_bench/question.jsonl",
              "--spec-bench-category", cat]
        elapsed, rc = _run_bench_p1(ds, out_dir, "run.json", concurrency=192)
        print(f"    [{arm}][phase5][{cat}] elapsed={elapsed:.1f}s rc={rc}", flush=True)


def cold_start_burn(arm):
    """Burn flashinfer JIT autotune with short synthetic load before real measurements.
    Critical: β v2 confirmed cold-start bias flipped seed 0 vs seed 1/2 throughput by ~25%.
    """
    burn_dir = RESULTS_DIR / "burn" / arm
    burn_dir.mkdir(parents=True, exist_ok=True)
    print(f"    [{arm}] cold-start burn (short random, c=64, 2 rounds)...", flush=True)
    for i in range(2):
        ds = ["--dataset-name", "random",
              "--random-input-len", "512",
              "--random-output-len", "64"]
        elapsed, rc = _run_bench_p1(ds, burn_dir, f"burn_{i}.json", concurrency=64)
        print(f"    [{arm}][burn_{i}] elapsed={elapsed:.1f}s rc={rc}", flush=True)


def run_arm(arm, phases):
    print(f"\n========== ARM: {arm} :: PHASES: {phases} ==========", flush=True)
    proc, log = launch_server(arm)
    try:
        # Phase 0 probe (idle metrics baseline snapshot)
        time.sleep(5)
        m0 = get_metrics()
        (RESULTS_DIR / f"phase0_{arm}_metrics.txt").write_text(m0)
        print(f"    [{arm}] Phase 0 metrics snapshot saved", flush=True)

        # Cold-start burn (β v2 lesson: seed 0 bias)
        cold_start_burn(arm)

        if "phase4" in phases:
            phase4_bench(arm)
        if "phase1" in phases:
            ds = os.environ.get("PHASE1_DATASET", "random")
            phase1_bench(arm, dataset=ds)
        if "phase5" in phases:
            phase5_bench(arm)
    finally:
        kill_server(proc, log)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    # arm_phase syntax: "A_prime:phase4,phase1" separated by comma
    if len(sys.argv) < 2:
        print("Usage: master.py <arm1:phases,arm2:phases,...>")
        print("Example: master.py 'A_prime:phase4 C_prime:phase4 no_spec:phase4 static_k3:phase4'")
        sys.exit(1)
    for spec in sys.argv[1:]:
        arm, phases_str = spec.split(":")
        phases = phases_str.split(",")
        try:
            run_arm(arm, phases)
        except Exception as e:
            print(f"    [{arm}] ERROR: {e}", flush=True)
            continue
    print("\n========== ALL DONE ==========", flush=True)


if __name__ == "__main__":
    main()
