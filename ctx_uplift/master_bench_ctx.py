#!/usr/bin/env python3
"""
ctx-uplift master bench (pre-reg: ctx_uplift/PREREGISTRATION.md, commit 825abd5).

Tests the cross-ctx contradiction: is optimal K different at 4k vs 32k for a
FIXED batch? One server per K (K is launch-time), each server benched at BOTH
ctx (ctx is bench-time = --prefix-repetition-prefix-len), so 6 servers cover the
12 spine cells. Adapted from tax_decomposition/master_bench_tax.py; only the
arm/K/ctx logic differs.

K semantics (pre-reg §4/§9):
  k0     = DSD-tier K=0: drafter loaded, single-tier schedule [[1,512,0]] forces
           K=0 at every batch. This is the "table assigns 0" path (#49986 tax).
  k1..k7 = fixed K via num_speculative_tokens=K, NO schedule key (schedule would
           override K). K>=5 requires the MTP drafter to support that many tokens
           — verified per-arm from the server SpeculativeConfig line + acceptance.
  no_spec = reference baseline (speculative_config absent); 4k anchor arm.

Build: match pr48944_replication / tax stack (pin SHA), but --max-model-len 33024
(required for 32k; deviation recorded). NO --kv-cache-dtype fp8 (#48495); KV=bf16.

Gate first (fixes b from the engine's own KV log line, authoritative over hand-calc):
  RESULTS_DIR=/root/results VLLM_REPO=/workspace/vllm python master_bench_ctx.py gate
Then set SPINE_B and run arms:
  SPINE_B=11 RESULTS_DIR=/root/results python master_bench_ctx.py k0 k1 k2 k3 k5 k7 no_spec
  SPINE_B=11 RESULTS_DIR=/root/results python master_bench_ctx.py spot
"""
import subprocess, json, time, sys, os, signal, shutil, re
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

MODEL = "/root/models/target"
DRAFT = "/root/models/draft"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
VLLM_REPO = os.environ.get("VLLM_REPO", "/workspace/vllm")
MAX_MODEL_LEN = "34816"          # 32768 prefix + 96 suffix + 256 out + BOS/template headroom
                                 # (pre-reg §9 said 33024=32768+256, too tight by BOS+suffix:
                                 #  requests hit HTTP400 input+output>mml; raised, ctx unchanged)
CTXS = [int(x) for x in os.environ.get("CTX_UPLIFT_CTXS", "4096,32768").split(",")]  # bench-time; both run on each server (env-override for precision re-run)
OUTPUT_LEN = 256                 # pre-reg §4: >=256 to keep decode window (not 100)
TIMEOUT = 900                    # 33024 KV alloc + FULL cudagraph capture is slow
NUM_PROMPT_ROUNDS = {4096: 16, 32768: 8}   # waves of `b`; 32k slower -> fewer

KS = [0, 1, 2, 3, 5, 7]
ARMS = [f"k{k}" for k in KS] + ["no_spec", "spot"]  # spot re-measures k0 at the end
WARMUP = int(os.environ.get("CTX_UPLIFT_WARMUP", "3"))  # discarded warmup rounds/ctx; raise for 32k spec steady-state (PREREGISTRATION_precision.md)


def spine_b():
    b = os.environ.get("SPINE_B")
    if not b:
        sys.exit("SPINE_B unset — run `gate` first, then export SPINE_B=<b>.")
    return int(b)


def _k_of(arm):
    if arm == "no_spec":
        return None
    if arm == "spot":
        return 0                 # spot mirrors the k0 reference
    return int(arm[1:])          # "k5" -> 5


def spec_config(arm):
    k = _k_of(arm)
    if k is None:
        return None
    cfg = {"model": DRAFT, "num_speculative_tokens": 3 if k == 0 else k}
    if k == 0:
        cfg["num_speculative_tokens_per_batch_size"] = [[1, 512, 0]]  # DSD-tier K=0
    # k>=1: fixed K, NO schedule key on purpose (a key here would override K).
    return cfg


def num_prompts_for(ctx, b):
    return max(b * NUM_PROMPT_ROUNDS[ctx], 32)


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


def launch_server(arm):
    wipe_autotune()
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", MAX_MODEL_LEN,
    ]
    cfg = spec_config(arm)
    if cfg is not None:
        cmd += ["--speculative-config", json.dumps(cfg)]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONHASHSEED"] = "0"
    log_path = f"/tmp/server_{arm}.log"
    log = open(log_path, "w")
    log.write(f"CMD: {' '.join(cmd)}\nspec_config={json.dumps(cfg)}\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    print(f"[{arm}] server PID={proc.pid}, waiting /health "
          f"(timeout={TIMEOUT}s)...", flush=True)
    t0, last = time.time(), 0
    while time.time() - t0 < TIMEOUT:
        elapsed = time.time() - t0
        if proc.poll() is not None:
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


def record_kv_capacity(arm):
    """Engine's own KV log lines. With mml=33024 and 32k req = 32768+256, the
    'Maximum concurrency' line IS the 32k reachable concurrency (authoritative
    over our ~12 hand-calc; the engine accounts the hybrid sliding cap itself)."""
    log = Path(f"/tmp/server_{arm}.log").read_text(errors="ignore")
    lines = [l.strip() for l in log.split("\n")
             if "KV cache size" in l or "Maximum concurrency" in l]
    (RESULTS_DIR / f"kv_capacity_{arm}.txt").write_text("\n".join(lines) + "\n")
    b_suggest = None
    for l in lines:
        print(f"    [{arm}][KV] {l}", flush=True)
        m = re.search(r"Maximum concurrency.*?:\s*([\d.]+)x", l)
        if m:
            b_suggest = int(float(m.group(1)) * 0.9)
    if b_suggest is not None:
        print(f"    [{arm}] suggested b = floor(0.9 * concurrency) = {b_suggest} "
              f"(hand-calc was ~12/11; record if far off)", flush=True)
    return b_suggest


def record_spec_config(arm):
    """Verify the ACTUAL K the engine used (K=5/7 may be capped by MTP heads)."""
    log = Path(f"/tmp/server_{arm}.log").read_text(errors="ignore")
    m = re.search(r"SpeculativeConfig\([^)]*\)", log)
    line = m.group(0) if m else "NONE"
    (RESULTS_DIR / f"spec_config_{arm}.txt").write_text(line + "\n")
    print(f"    [{arm}] actual spec config: {line}", flush=True)


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
        if (line.startswith(name) and "{" in line and "}" in line) or \
           line.startswith(name + " "):
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


def run_bench(ctx, b, out_dir, out_file):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = num_prompts_for(ctx, b)
    cmd = [
        "vllm", "bench", "serve",
        "--model", MODEL,
        "--port", str(PORT),
        "--num-prompts", str(n_prompts),
        "--max-concurrency", str(b),
        "--ignore-eos",
        "--dataset-name", "prefix_repetition",
        "--prefix-repetition-prefix-len", str(ctx),
        "--prefix-repetition-suffix-len", "96",
        "--prefix-repetition-num-prefixes", "1",
        "--prefix-repetition-output-len", str(OUTPUT_LEN),
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
    return elapsed, r.returncode, n_prompts


def cold_start_burn(arm, b):
    burn_dir = RESULTS_DIR / "burn" / arm
    burn_dir.mkdir(parents=True, exist_ok=True)
    print(f"    [{arm}] cold-start burn (c={b}, 2 rounds)...", flush=True)
    for i in range(2):
        cmd = [
            "vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT),
            "--num-prompts", str(max(b * 4, 32)), "--max-concurrency", str(b),
            "--ignore-eos", "--dataset-name", "random",
            "--random-input-len", "512", "--random-output-len", "64",
            "--save-result", "--result-dir", str(burn_dir),
            "--result-filename", f"burn_{i}.json",
        ]
        env = os.environ.copy()
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    time.sleep(30)


def phase_grid(arm, b):
    """Both ctx on this server; 3 warmup discarded + 3 measured per ctx."""
    for ctx in CTXS:
        out_dir = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(WARMUP + 3):
            is_measure = i >= WARMUP
            label = "measure" if is_measure else "warmup"
            seed = i - WARMUP if is_measure else i
            fname = f"{label}_{seed}.json"
            m0 = get_metrics()
            elapsed, rc, n_prompts = run_bench(ctx, b, out_dir, fname)
            m1 = get_metrics()
            if m0 and m1:
                (out_dir / f"snapshot_{label}_{seed}.json").write_text(
                    json.dumps(snapshot_delta(m0, m1, elapsed), indent=2))
            print(f"    [{arm}][ctx={ctx}][{label}_{seed}] "
                  f"elapsed={elapsed:.1f}s rc={rc} num_prompts={n_prompts}",
                  flush=True)


def run_arm(arm, b):
    print(f"\n========== ARM: {arm} (b={b}) ==========", flush=True)
    record_build_sha(arm)
    grid_arm = "spot" if arm == "spot" else arm
    proc, log = launch_server(arm)
    try:
        time.sleep(5)
        (RESULTS_DIR / f"phase0_{arm}_metrics.txt").write_text(get_metrics())
        record_kv_capacity(arm)
        record_spec_config(arm)
        cold_start_burn(arm, b)
        phase_grid(grid_arm, b)
    finally:
        kill_server(proc, log)


def run_gate():
    """Launch one spec-on server (k1, so draft VRAM is reflected), read the KV
    capacity line, suggest b, and stop. No bench — this is the pre-sweep gate."""
    print("\n========== GATE (fix b from engine KV log) ==========", flush=True)
    proc, log = launch_server("k1")
    try:
        time.sleep(5)
        record_kv_capacity("k1")
        record_spec_config("k1")
        print("    GATE: export SPINE_B=<suggested b>, then run the arms.",
              flush=True)
    finally:
        kill_server(proc, log)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2:
        print("Usage: master_bench_ctx.py gate | <arm> [<arm> ...]")
        print("arms:", ", ".join(ARMS))
        sys.exit(1)
    if sys.argv[1] == "gate":
        run_gate()
        return
    b = spine_b()
    for arm in sys.argv[1:]:
        if arm not in ARMS:
            print(f"    unknown arm '{arm}', skipping", flush=True)
            continue
        try:
            run_arm(arm, b)
        except Exception as e:
            print(f"    [{arm}] ERROR: {e}", flush=True)
            continue
    print("\n========== ALL DONE ==========", flush=True)


if __name__ == "__main__":
    main()
