#!/usr/bin/env python3
"""
2x2 attribution bench (pre-reg: tax_attribution/PREREGISTRATION.md).

Splits the DSD baseline tax between the cudagraph term and the drafter-KV term
by varying graph mode and KV pool size independently on the no_spec arm, with
dsd_k0 as the reference cell.

Usage:
  RESULTS_DIR=/root/results python master_bench_attrib.py --probe
  RESULTS_DIR=/root/results KV_HIGH_MIB=48128 KV_LOW_MIB=43038 python master_bench_attrib.py --paired
"""
import argparse, json, os, re, shutil, signal, subprocess, time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

MODEL = os.environ.get("MODEL", "/root/models/target")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "/root/models/draft")
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/root/results"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "52224"))
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "512"))
MIB = 1024 * 1024

# Set on the rental from --probe output. None means "let vLLM profile it",
# which reintroduces the launch-order offset and is only for the probe pass.
KV_HIGH_MIB = int(os.environ.get("KV_HIGH_MIB", "0")) or None
KV_LOW_MIB = int(os.environ.get("KV_LOW_MIB", "0")) or None

CTX = int(os.environ.get("CTX", "400"))
CONCURRENCIES = [int(c) for c in os.environ.get("CONCURRENCIES", "189,2").split(",")]
SUFFIX_TOKENS = 197
POOL_MARGIN = 0.9

DSD_K0_SCHEDULE = [[1, 512, 0]]

ARMS = {
    "base_full_high":  {"spec": False, "cudagraph": None,        "pool": "high"},
    "base_piece_high": {"spec": False, "cudagraph": "PIECEWISE", "pool": "high"},
    "base_full_low":   {"spec": False, "cudagraph": None,        "pool": "low"},
    "base_piece_low":  {"spec": False, "cudagraph": "PIECEWISE", "pool": "low"},
    "dsd_k0_low":      {"spec": True,  "cudagraph": None,        "pool": "low"},
}
ARM_ORDER = ["base_full_high", "base_piece_high", "base_full_low", "base_piece_low", "dsd_k0_low"]

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

POOL_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
MEM_RE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
CAPTURE_RE = re.compile(r"Capturing CUDA graphs \((?:mixed prefill-decode|decode), (\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)")
MODE_RE = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)")
COMPCFG_RE = re.compile(r"compilation_config=(\{.*?\}), kernel_config=")
VOLATILE_KEYS = {"cudagraph_mode": r"<[^>]*>",
                 "cache_dir": r"[^,}]*",
                 "local_cache_dir": r"[^,}]*",
                 "debug_dump_path": r"[^,}]*"}
KV_PIN_SAFETY_MIB = int(os.environ.get("KV_PIN_SAFETY_MIB", "128"))


def pools_path():
    return RESULTS_DIR / "pools.json"


def load_pools():
    p = pools_path()
    return json.loads(p.read_text()) if p.exists() else []


def record_launch(arm, tag, tokens, gib, discarded):
    entries = load_pools()
    index = sum(1 for e in entries if e["arm"] == arm)
    entries.append({"arm": arm, "tag": tag, "launch_index": index,
                    "tokens": tokens, "gib": gib, "discarded": discarded})
    pools_path().write_text(json.dumps(entries, indent=2))
    return index


def parse_pool(tag):
    text = Path(f"/tmp/server_{tag}.log").read_text(errors="ignore")
    tok = POOL_RE.findall(text)
    mem = MEM_RE.findall(text)
    tokens = int(tok[-1].replace(",", "")) if tok else None
    gib = float(mem[-1]) if mem else None
    return tokens, gib


def assert_pools_comparable(arm_a, arm_b):
    for arm in (arm_a, arm_b):
        entries = [e for e in load_pools() if e["arm"] == arm]
        if not entries:
            raise SystemExit(f"no launch recorded for arm {arm}")
        if not any(e["discarded"] for e in entries):
            raise SystemExit(
                f"arm {arm} has no discarded first launch; its pool carries the "
                f"launch-order offset and cannot be compared across arms"
            )
        if not any(not e["discarded"] for e in entries):
            raise SystemExit(f"arm {arm} has only a discarded launch")


def kv_bytes_for(arm):
    want = ARMS[arm]["pool"]
    mib = KV_HIGH_MIB if want == "high" else KV_LOW_MIB
    return mib * MIB if mib else None


def wipe_autotune():
    shutil.rmtree(Path.home() / ".cache/vllm/flashinfer_autotune_cache", ignore_errors=True)


def assert_arm_is_k0(concurrency):
    for lo, hi, k in DSD_K0_SCHEDULE:
        if lo <= concurrency <= hi:
            if k != 0:
                raise SystemExit(
                    f"arm dsd_k0_low would run at K={k} for concurrency={concurrency}")
            return
    raise SystemExit(f"concurrency={concurrency} falls outside {DSD_K0_SCHEDULE}")


def assert_concurrency_fits(concurrency, pool_tokens):
    need = concurrency * (CTX + SUFFIX_TOKENS)
    cap = POOL_MARGIN * pool_tokens
    if need > cap:
        raise SystemExit(
            f"concurrency={concurrency} needs {need} tokens but the pool cap is "
            f"{cap:.0f} ({pool_tokens} x {POOL_MARGIN}); the cell would preempt")


def spec_config(arm):
    if not ARMS[arm]["spec"]:
        return None
    return {"model": DRAFT_MODEL, "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": DSD_K0_SCHEDULE}


def launch_server(arm, tag):
    wipe_autotune()
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT),
           "--gpu-memory-utilization", "0.90",
           "--max-model-len", str(MAX_MODEL_LEN), "--enable-prefix-caching"]
    kv_bytes = kv_bytes_for(arm)
    if kv_bytes:
        cmd += ["--kv-cache-memory-bytes", str(kv_bytes)]
    cg = ARMS[arm]["cudagraph"]
    if cg:
        cmd += ["--compilation-config", json.dumps({"cudagraph_mode": cg})]
    cfg = spec_config(arm)
    if cfg is not None:
        cmd += ["--speculative-config", json.dumps(cfg)]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    log_path = f"/tmp/server_{tag}.log"
    log = open(log_path, "w")
    log.write(f"CMD: {' '.join(cmd)}\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    print(f"[{tag}] server PID={proc.pid}, waiting /health...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 900:
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


def discard_first_launch(arm):
    """Burn one launch per arm so later launches sit in the box's steady state."""
    tag = f"{arm}_discard"
    proc, log = launch_server(arm, tag)
    tokens, gib = parse_pool(tag)
    kill_server(proc, log)
    shutil.copy(f"/tmp/server_{tag}.log", RESULTS_DIR / f"server_{tag}.log")
    record_launch(arm, tag, tokens, gib, discarded=True)
    print(f"[{arm}] discarded first launch: {tokens} tokens / {gib} GiB", flush=True)
    return tokens, gib


def parse_captures(tag):
    text = Path(f"/tmp/server_{tag}.log").read_text(errors="ignore")
    counts = {}
    for mode, done, total in CAPTURE_RE.findall(text):
        if done == total:
            counts[mode] = int(total)
    declared = MODE_RE.findall(text)
    return counts, (declared[-1] if declared else None)


def assert_graph_mode(arm, tag):
    """The 2x2 is meaningless unless the intended graph mode actually took."""
    counts, declared = parse_captures(tag)
    want_full = ARMS[arm]["cudagraph"] is None and not ARMS[arm]["spec"]
    got_full = counts.get("FULL", 0) > 0
    if want_full and not got_full:
        raise SystemExit(
            f"arm {arm} was meant to keep FULL graphs but captured none "
            f"(declared={declared}, counts={counts})")
    if not want_full and got_full:
        raise SystemExit(
            f"arm {arm} was meant to run without FULL graphs but captured "
            f"{counts['FULL']} (declared={declared}, counts={counts})")
    (RESULTS_DIR / f"captures_{tag}.json").write_text(
        json.dumps({"declared_mode": declared, "captures": counts}, indent=2))
    print(f"[{tag}] cudagraph {declared} captures={counts}", flush=True)


def compilation_fingerprint(tag):
    """The compilation config with the keys this study varies masked out."""
    text = Path(f"/tmp/server_{tag}.log").read_text(errors="ignore")
    m = COMPCFG_RE.search(text)
    if not m:
        return None
    cfg = m.group(1)
    for key, pattern in VOLATILE_KEYS.items():
        cfg = re.sub(rf"'{key}': {pattern}", f"'{key}': <masked>", cfg)
    return cfg


def assert_compilation_matches(arm, tag):
    """Forcing cudagraph_mode must change only cudagraph_mode.

    --compilation-config is passed as JSON, so a merge that turned out to be a
    replace would silently move splitting_ops or the capture sizes underneath
    the forced-PIECEWISE arms and the 2x2 would be comparing two different
    baselines.
    """
    fp = compilation_fingerprint(tag)
    ref_path = RESULTS_DIR / "compilation_fingerprint.txt"
    (RESULTS_DIR / f"compcfg_{tag}.txt").write_text(fp or "<unparsed>")
    if fp is None:
        raise SystemExit(f"could not parse compilation_config for {tag}")
    if not ref_path.exists():
        ref_path.write_text(fp)
        return
    ref = ref_path.read_text()
    if fp != ref:
        (RESULTS_DIR / f"compcfg_mismatch_{tag}.txt").write_text(
            f"reference:\n{ref}\n\nthis arm ({arm}):\n{fp}\n")
        raise SystemExit(
            f"arm {arm} differs from the reference compilation config in more "
            f"than {sorted(VOLATILE_KEYS)}; see compcfg_mismatch_{tag}.txt")


def assert_pinned(arm, tag, tokens):
    if kv_bytes_for(arm) is None:
        return
    prev = [e for e in load_pools()
            if e["arm"] == arm and not e["discarded"] and e["tag"] != tag]
    if prev and prev[-1]["tokens"] != tokens:
        raise SystemExit(
            f"arm {arm} pool drifted between pinned launches: "
            f"{prev[-1]['tokens']} then {tokens}; --kv-cache-memory-bytes did not hold")


def get_metrics():
    try:
        with urlopen(f"{BASE_URL}/metrics", timeout=10) as r:
            return r.read().decode()
    except Exception:
        return None


def scrape(text, name):
    total = 0.0
    for line in text.split("\n"):
        if line.startswith(name) and "_bucket" not in line:
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return total


def snapshot_delta(m0, m1, elapsed):
    return {"elapsed_s": elapsed,
            **{f"{m}_delta": scrape(m1, m) - scrape(m0, m) for m in DELTA_METRICS}}


def run_bench(arm, concurrency, out_dir, out_file):
    if ARMS[arm]["spec"]:
        assert_arm_is_k0(concurrency)
    cmd = ["vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT),
           "--num-prompts", str(NUM_PROMPTS), "--max-concurrency", str(concurrency),
           "--ignore-eos", "--dataset-name", "prefix_repetition",
           "--prefix-repetition-prefix-len", str(CTX),
           "--prefix-repetition-suffix-len", "96",
           "--prefix-repetition-num-prefixes", "1",
           "--prefix-repetition-output-len", "100",
           "--percentile-metrics", "ttft,tpot,itl",
           "--save-result", "--result-dir", str(out_dir), "--result-filename", out_file]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        (out_dir / f"{out_file}.stderr").write_text(r.stderr[-4000:])
    return time.time() - t0, r.returncode


def cold_start_burn():
    burn_dir = RESULTS_DIR / "burn"
    burn_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        subprocess.run(
            ["vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT),
             "--num-prompts", "64", "--max-concurrency", "64", "--ignore-eos",
             "--dataset-name", "random", "--random-input-len", "512",
             "--random-output-len", "64", "--save-result",
             "--result-dir", str(burn_dir), "--result-filename", f"burn_{i}.json"],
            capture_output=True, text=True, timeout=1800)
    time.sleep(30)


def run_cell(arm, concurrency):
    tag = f"{arm}_c{concurrency}"
    proc, log = launch_server(arm, tag)
    try:
        tokens, gib = parse_pool(tag)
        assert_pinned(arm, tag, tokens)
        record_launch(arm, tag, tokens, gib, discarded=False)
        assert_concurrency_fits(concurrency, tokens)
        assert_graph_mode(arm, tag)
        assert_compilation_matches(arm, tag)
        cold_start_burn()
        out_dir = RESULTS_DIR / "grid" / arm / f"ctx_{CTX}_c{concurrency}"
        out_dir.mkdir(parents=True, exist_ok=True)
        reps = int(os.environ.get("REPS", "4"))
        n_warm = reps - 3
        for i in range(reps):
            is_measure = i >= n_warm
            label = "measure" if is_measure else "warmup"
            seed = i - n_warm if is_measure else i
            m0 = get_metrics()
            elapsed, rc = run_bench(arm, concurrency, out_dir, f"{label}_{seed}.json")
            m1 = get_metrics()
            if m0 and m1:
                (out_dir / f"snapshot_{label}_{seed}.json").write_text(
                    json.dumps(snapshot_delta(m0, m1, elapsed), indent=2))
            print(f"    [{arm}][c={concurrency}][{label}_{seed}] "
                  f"elapsed={elapsed:.1f}s rc={rc}", flush=True)
    finally:
        shutil.copy(f"/tmp/server_{tag}.log", RESULTS_DIR / f"server_{tag}.log")
        kill_server(proc, log)


def probe():
    """Measure each arm's steady-state pool, discarding its first launch."""
    if KV_HIGH_MIB or KV_LOW_MIB:
        raise SystemExit("run --probe with KV_HIGH_MIB and KV_LOW_MIB unset")
    out = {}
    for arm in ("base_full_high", "dsd_k0_low"):
        discard_first_launch(arm)
        tag = f"{arm}_probe"
        proc, log = launch_server(arm, tag)
        tokens, gib = parse_pool(tag)
        kill_server(proc, log)
        shutil.copy(f"/tmp/server_{tag}.log", RESULTS_DIR / f"server_{tag}.log")
        record_launch(arm, tag, tokens, gib, discarded=False)
        out[arm] = {"tokens": tokens, "gib": gib}
        print(f"[{arm}] steady pool: {tokens} tokens / {gib} GiB", flush=True)
    hi, lo = out["base_full_high"], out["dsd_k0_low"]
    for name, v in (("KV_HIGH_MIB", hi), ("KV_LOW_MIB", lo)):
        print(f"{name}={int(v['gib'] * 1024) - KV_PIN_SAFETY_MIB}", flush=True)
    delta = hi["tokens"] - lo["tokens"]
    print(f"drafter KV cost: {delta} tokens "
          f"({100 * delta / hi['tokens']:.2f}% of the no-spec pool)", flush=True)
    (RESULTS_DIR / "PROBE.json").write_text(json.dumps(out, indent=2))


def run_paired():
    if not (KV_HIGH_MIB and KV_LOW_MIB):
        raise SystemExit("set KV_HIGH_MIB and KV_LOW_MIB from --probe before measuring")
    done = []
    for concurrency in CONCURRENCIES:
        print(f"\n===== concurrency={concurrency} =====", flush=True)
        for arm in ARM_ORDER:
            if not any(e["arm"] == arm and e["discarded"] for e in load_pools()):
                discard_first_launch(arm)
            run_cell(arm, concurrency)
        done.append(concurrency)
        (RESULTS_DIR / "CELLS_DONE.txt").write_text(
            "\n".join(str(c) for c in done) + "\n")
        print(f"===== concurrency={concurrency} COMPLETE (done: {done}) =====", flush=True)
    assert_pools_comparable("base_full_high", "dsd_k0_low")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=[], choices=ARM_ORDER + [])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--conc", type=int, default=0)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe()
        return
    if args.paired:
        run_paired()
        return
    for arm in args.arms:
        if not any(e["arm"] == arm and e["discarded"] for e in load_pools()):
            discard_first_launch(arm)
        for c in ([args.conc] if args.conc else CONCURRENCIES):
            run_cell(arm, c)


if __name__ == "__main__":
    main()
