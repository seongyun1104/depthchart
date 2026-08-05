#!/usr/bin/env python3
"""
Tax-decomposition master bench (pre-reg v3: tax_decomposition/PREREGISTRATION.md).

Isolates the graph-mode (V1 PIECEWISE -> V2 FULL) component of the #49986 DSD
tax on the ORIGINAL stack (Gemma-4-31B hybrid + MTP), toggling ONLY the runner
via VLLM_USE_V2_MODEL_RUNNER. Adapted from pr48944_replication/master_bench.py;
only the arm/schedule/runner-toggle logic differs.

Build: #49652 @ fd355781 for arms A/B/C/D; upstream main for the spot check S.
  Run A/B/C/D on the branch build, then rebuild main and run `spot`.

#49986 stack reproduced verbatim from pr49986_runbook/RUNBOOK.md (grep-verified
2026-08-05): num_speculative_tokens=3, --ignore-eos present,
--gpu-memory-utilization 0.90 --max-model-len 8192, NO --kv-cache-dtype fp8.
KV pool at these settings is 55,215 tokens (log-measured), so ctx=4000 runs at
concurrency 192 (>=129 keeps K=0 tier) to stay off the preemption margin.

Usage:
  RESULTS_DIR=/root/results VLLM_REPO=/workspace/vllm \\
      python master_bench_tax.py v1_no_spec v1_dsd_k0
  RESULTS_DIR=/root/results python master_bench_tax.py v2_no_spec v2_dsd_k0
  # after rebuilding upstream main:
  RESULTS_DIR=/root/results python master_bench_tax.py spot
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
VLLM_REPO = os.environ.get("VLLM_REPO", "/workspace/vllm")

# arm -> (use_v2, spec_on). S ("spot") == v1_dsd_k0 but run on the main build.
ARMS = {
    "v1_no_spec": (False, False),  # A
    "v1_dsd_k0":  (False, True),   # B
    "v2_no_spec": (True,  False),  # C
    "v2_dsd_k0":  (True,  True),   # D
    "spot":       (False, True),   # S (main build, V1, dsd-K0; == #49986 point)
}
# bs>=129 -> K=0, so both concurrencies below land the workload in the K=0 tier.
DSD_K0_SCHEDULE = [[1, 64, 3], [65, 128, 0], [129, 512, 0]]
# Per-ctx concurrency (pre-reg §5.3): 4000 at 192 keeps KV off the ~1k margin.
# Applied uniformly across all 4 arms at each ctx, else within-ctx tax is invalid.
CTX_CONCURRENCY = {400: 256, 4000: 192}
NUM_PROMPTS = 1024


def wipe_autotune():
    p = Path.home() / ".cache/vllm/flashinfer_autotune_cache"
    existed = p.exists()
    shutil.rmtree(p, ignore_errors=True)
    print(f"    autotune wipe: existed={existed} ({p})", flush=True)


def record_build_sha(arm):
    try:
        sha = subprocess.check_output(
            ["git", "-C", VLLM_REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:
        sha = f"unknown ({e})"
    (RESULTS_DIR / f"build_sha_{arm}.txt").write_text(sha + "\n")
    print(f"    [{arm}] build SHA {sha[:12]}", flush=True)


def spec_config(arm):
    if not ARMS[arm][1]:
        return None
    return {
        "model": DRAFT,
        "num_speculative_tokens": 3,
        "num_speculative_tokens_per_batch_size": DSD_K0_SCHEDULE,
    }


def launch_server(arm):
    wipe_autotune()
    use_v2 = ARMS[arm][0]
    timeout = 1200 if use_v2 else 600  # FULL cudagraph capture is slower than PIECEWISE
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192",
    ]
    cfg = spec_config(arm)
    if cfg is not None:
        cmd += ["--speculative-config", json.dumps(cfg)]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if use_v2 else "0"  # the single lever
    log_path = f"/tmp/server_{arm}.log"
    log = open(log_path, "w")
    log.write(f"CMD: {' '.join(cmd)}\nVLLM_USE_V2_MODEL_RUNNER="
              f"{env['VLLM_USE_V2_MODEL_RUNNER']}\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    print(f"[{arm}] server PID={proc.pid} v2={use_v2}, waiting /health "
          f"(timeout={timeout}s)...", flush=True)
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout:
        elapsed = time.time() - t0
        if proc.poll() is not None:  # process died: stop polling immediately
            print(f"[{arm}] process exited early rc={proc.returncode} "
                  f"after {elapsed:.1f}s", flush=True)
            break
        try:
            with urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[{arm}] READY after {elapsed:.1f}s", flush=True)
                    return proc, log
        except (URLError, ConnectionError, OSError):
            pass
        if elapsed - last >= 30:
            print(f"    [{arm}] waiting... {elapsed:.0f}s", flush=True)
            last = elapsed
        time.sleep(5)
    kill_server(proc, log)
    raise RuntimeError(f"Server {arm} failed to start (see {log_path})")


def gate_b_check(arm):
    """Record V2 launch outcome from the server log. Called on success AND on
    launch failure (the log is on disk either way) — Gate B FAIL is the primary
    #49652 deliverable, so it must leave structured data.
    """
    if not ARMS[arm][0]:
        return
    log = Path(f"/tmp/server_{arm}.log").read_text(errors="ignore")
    assert_hit = "0 < num_reqs <= num_tokens" in log or "make_dummy" in log
    full_cg = "FULL cudagraph" in log or "FULL_AND_PIECEWISE" in log
    verdict = {"arm": arm, "v48494_assert_seen": assert_hit,
               "full_cudagraph_seen": full_cg, "log_tail": log[-2000:]}
    (RESULTS_DIR / f"gate_b_{arm}.json").write_text(json.dumps(verdict, indent=2))
    print(f"    [GATE B][{arm}] assert={assert_hit} full_cg={full_cg}", flush=True)


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
    print("    server killed, waiting KV free...", flush=True)
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
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
    ]
    d = {"elapsed_sec": elapsed}
    for m in metrics:
        d[m.replace("vllm:", "").replace("_total", "_delta")] = (
            parse_metric(m_after, m) - parse_metric(m_before, m)
        )
    return d


def run_bench(ctx, out_dir, out_file):
    out_dir.mkdir(parents=True, exist_ok=True)
    concurrency = CTX_CONCURRENCY[ctx]
    cmd = [
        "vllm", "bench", "serve",
        "--model", MODEL,
        "--port", str(PORT),
        "--num-prompts", str(NUM_PROMPTS),
        "--max-concurrency", str(concurrency),
        "--ignore-eos",
        "--dataset-name", "prefix_repetition",
        "--prefix-repetition-prefix-len", str(ctx),
        "--prefix-repetition-suffix-len", "96",
        "--prefix-repetition-num-prefixes", "1",
        "--prefix-repetition-output-len", "100",
        "--percentile-metrics", "ttft,tpot,itl",
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", out_file,
    ]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    if r.returncode != 0:
        (out_dir / f"{out_file}.stderr").write_text(r.stderr[-4000:])
    return elapsed, r.returncode


def cold_start_burn(arm):
    """2 short rounds (c=64) to settle flashinfer JIT autotune + CG capture.
    Repeated per server restart (β v2 lesson: seed-0 throughput bias ~25%).
    """
    burn_dir = RESULTS_DIR / "burn" / arm
    burn_dir.mkdir(parents=True, exist_ok=True)
    print(f"    [{arm}] cold-start burn (c=64, 2 rounds)...", flush=True)
    for i in range(2):
        cmd = [
            "vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT),
            "--num-prompts", "64", "--max-concurrency", "64", "--ignore-eos",
            "--dataset-name", "random",
            "--random-input-len", "512", "--random-output-len", "64",
            "--save-result", "--result-dir", str(burn_dir),
            "--result-filename", f"burn_{i}.json",
        ]
        env = os.environ.copy()
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    time.sleep(30)


def phase_grid(arm):
    """ctx sweep, 3 warmup discarded + 3 measured. Metrics snapshot per run."""
    for ctx in CTX_CONCURRENCY:
        out_dir = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            is_measure = i >= 3
            label = "measure" if is_measure else "warmup"
            seed = i - 3 if is_measure else i
            fname = f"{label}_{seed}.json"
            m0 = get_metrics()
            elapsed, rc = run_bench(ctx, out_dir, fname)
            m1 = get_metrics()
            if m0 and m1:
                snap = snapshot_delta(m0, m1, elapsed)
                (out_dir / f"snapshot_{label}_{seed}.json").write_text(
                    json.dumps(snap, indent=2))
            print(f"    [{arm}][ctx={ctx}][{label}_{seed}] "
                  f"elapsed={elapsed:.1f}s rc={rc}", flush=True)


def run_arm(arm):
    print(f"\n========== ARM: {arm} ==========", flush=True)
    record_build_sha(arm)
    try:
        proc, log = launch_server(arm)
    except RuntimeError:
        gate_b_check(arm)  # capture the failure log as structured data
        raise
    try:
        time.sleep(5)
        (RESULTS_DIR / f"phase0_{arm}_metrics.txt").write_text(get_metrics())
        gate_b_check(arm)
        cold_start_burn(arm)
        phase_grid(arm)
    finally:
        kill_server(proc, log)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2:
        print("Usage: master_bench_tax.py <arm> [<arm> ...]")
        print("arms:", ", ".join(ARMS))
        sys.exit(1)
    for arm in sys.argv[1:]:
        if arm not in ARMS:
            print(f"    unknown arm '{arm}', skipping", flush=True)
            continue
        try:
            run_arm(arm)
        except Exception as e:
            print(f"    [{arm}] ERROR: {e}", flush=True)
            continue
    print("\n========== ALL DONE ==========", flush=True)


if __name__ == "__main__":
    main()
