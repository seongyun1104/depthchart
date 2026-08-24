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
NSYS = os.environ.get("NSYS", "0") == "1"
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "512"))

POOL_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
MEM_RE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
CAPTURE_RE = re.compile(
    r"Capturing CUDA graphs \((?:mixed prefill-decode|decode), (\w+)\):"
    r"\s*100%\|[^|]*\|\s*(\d+)/(\d+)")
MODE_RE = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)")

ARMS = {"no_spec": False, "dsd_k0": True}
# Arm B must mean the same thing at every ctx: "DSD enabled, drafting nothing".
# A tiered schedule cannot do that here -- the KV pool forces concurrency down as
# ctx grows (129 requests at ctx 16k would need ~2.1M pool tokens, far beyond a
# single H100), so a schedule whose K=0 tier starts at batch 129 would silently
# read K=3 at the long-ctx end and K=0 at the short end. The primary endpoint
# Tax(49400)-Tax(400) would then mix a ctx effect with a K 0->3 switch. Flat K=0
# keeps the arm constant so ctx is the only thing that varies.
DSD_K0_SCHEDULE = [[1, 512, 0]]

# Per-ctx concurrency: SET ON RENTAL from the measured KV pool (RUNBOOK §KV).
# The values below are placeholders scaled ~1/ctx off the tax_decomposition anchor.
# The only hard constraint is the preemption margin (see assert_arm_is_k0 for the
# schedule-side invariant): concurrency * (ctx + suffix) < 0.9 * pool, applied
# identically to both arms at each ctx.
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

# Per-step token census. vLLM exports this as a Prometheus histogram, so the
# bucket deltas around a run give the distribution of tokens-per-engine-step
# directly: decode-only steps land at ~1 token per running request, while
# chunked-prefill steps land in the high buckets. This replaces the DEBUG
# scheduler-log census -- current vLLM emits no per-step scheduler DEBUG line
# (verified against origin/main e25c586b90), so that parser matched nothing,
# and running the server at DEBUG to feed it would have taxed the very timing
# this study measures.
HISTOGRAM_METRICS = [
    "vllm:iteration_tokens_total",
]


def pools_path():
    return RESULTS_DIR / "pools.json"


def load_pools():
    return json.loads(pools_path().read_text()) if pools_path().exists() else []


def record_launch(arm, tag, tokens, gib, discarded):
    entries = load_pools()
    entries.append({"arm": arm, "tag": tag,
                    "launch_index": sum(1 for e in entries if e["arm"] == arm),
                    "tokens": tokens, "gib": gib, "discarded": discarded})
    pools_path().write_text(json.dumps(entries, indent=2))


def parse_pool(tag):
    text = Path(f"/tmp/server_{tag}.log").read_text(errors="ignore")
    tok = POOL_RE.findall(text)
    mem = MEM_RE.findall(text)
    return (int(tok[-1].replace(",", "")) if tok else None,
            float(mem[-1]) if mem else None)


def parse_captures(tag):
    text = Path(f"/tmp/server_{tag}.log").read_text(errors="ignore")
    counts = {m: int(tot) for m, done, tot in CAPTURE_RE.findall(text) if done == tot}
    declared = MODE_RE.findall(text)
    return counts, (declared[-1] if declared else None)


def assert_pools_comparable(arm_a, arm_b):
    """Refuse to compare pools measured at different launch positions.

    The first launch of an arm profiles ~0.53 GiB less KV memory than every
    launch after it -- 1586-1587 tokens, the same offset in both arms, stable to
    the token thereafter. Pairing across that offset is what made the first
    published drafter-KV figure 1 pp low.
    """
    for arm in (arm_a, arm_b):
        entries = [e for e in load_pools() if e["arm"] == arm]
        if not entries:
            raise SystemExit(f"no launch recorded for arm {arm}")
        if not any(e["discarded"] for e in entries):
            raise SystemExit(
                f"arm {arm} has no discarded first launch; its pool carries the "
                f"launch-order offset and is not comparable across arms")
        kept = [e for e in entries if not e["discarded"]]
        if not kept:
            raise SystemExit(f"arm {arm} has only a discarded launch")
        pools = {e["tokens"] for e in kept}
        if len(pools) > 1:
            raise SystemExit(
                f"arm {arm} reported more than one pool across measured "
                f"launches ({sorted(pools)}); the offset is not settled")


def wipe_autotune():
    p = Path.home() / ".cache/vllm/flashinfer_autotune_cache"
    shutil.rmtree(p, ignore_errors=True)


def assert_arm_is_k0(concurrency):
    """Fail loudly if the schedule would draft at this concurrency.

    Arm B is only interpretable as "the cost of having DSD on" while K is 0 at
    the batch size actually used. This is asserted per ctx rather than trusted,
    because the KV pool -- not the plan -- decides the concurrency.
    """
    for lo, hi, k in DSD_K0_SCHEDULE:
        if lo <= concurrency <= hi:
            if k != 0:
                raise SystemExit(
                    f"arm dsd_k0 would run at K={k} for concurrency={concurrency}; "
                    f"the ctx sweep would confound ctx with the K tier. "
                    f"Fix DSD_K0_SCHEDULE or the concurrency."
                )
            return
    raise SystemExit(
        f"concurrency={concurrency} falls outside DSD_K0_SCHEDULE ranges "
        f"{DSD_K0_SCHEDULE}; K would be undefined for arm dsd_k0."
    )


def spec_config(arm):
    if not ARMS[arm]:
        return None
    # Same shape as tax_decomposition/master_bench_tax.py::spec_config -- a
    # separate drafter model, no "method" key (vLLM infers draft_model from the
    # presence of "model"). The harness previously asked for "method": "mtp"
    # with no drafter, which cannot work: the target checkpoint carries no MTP
    # head (no mtp/nextn key in its config.json), and Gemma-4 MTP is a
    # Model-Runner-V2 feature besides -- V2 is exactly where K=0 is ignored
    # (#51510), which this study must avoid. Keeping the tax_decomposition form
    # is also what preserves comparability with its V1 +7.29% / V2 +16.64%.
    return {
        "model": DRAFT_MODEL,
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
    # No DEBUG logging: the per-step scheduler census it used to feed no longer
    # exists upstream, and DEBUG-level logging would tax the timing under study.
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
    counts, declared = parse_captures(tag)
    (RESULTS_DIR / f"captures_{tag}.json").write_text(
        json.dumps({"declared_mode": declared, "captures": counts}, indent=2))
    print(f"[{tag}] cudagraph {declared} captures={counts}", flush=True)


def discard_first_launch(arm):
    """Burn one launch per arm so measured launches sit in the box steady state."""
    tag = f"{arm}_discard"
    proc, log = launch_server(arm, tag)
    tokens, gib = parse_pool(tag)
    kill_server(proc, log)
    shutil.copy(f"/tmp/server_{tag}.log", RESULTS_DIR / f"server_{tag}.log")
    record_launch(arm, tag, tokens, gib, discarded=True)
    print(f"[{arm}] discarded first launch: {tokens} tokens / {gib} GiB", flush=True)


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


def parse_histogram(text, name):
    """Return {le_bound: cumulative_count} for a Prometheus histogram."""
    buckets = {}
    prefix = name + "_bucket{"
    for line in text.split("\n"):
        if not line.startswith(prefix):
            continue
        try:
            le = line.split('le="', 1)[1].split('"', 1)[0]
            buckets[le] = buckets.get(le, 0.0) + float(line.split()[-1])
        except (ValueError, IndexError):
            pass
    return buckets


def histogram_delta(h0, h1):
    """Cumulative bucket deltas -> per-bucket step counts (non-cumulative)."""
    def _key(le):
        return float("inf") if le == "+Inf" else float(le)

    delta = {le: h1.get(le, 0.0) - h0.get(le, 0.0) for le in h1}
    ordered = sorted(delta, key=_key)
    out, prev = {}, 0.0
    for le in ordered:
        out[le] = delta[le] - prev
        prev = delta[le]
    return out


def census_split(hist_delta, concurrency):
    """Split steps into decode-only vs prefill-heavy.

    A decode-only step schedules ~1 token per running request, so it cannot
    exceed the concurrency ceiling; anything above that bucket carries a
    prefill chunk. Reported alongside the raw buckets -- the threshold is a
    reading aid, not a measurement.
    """
    decode = prefill = 0.0
    for le, n in hist_delta.items():
        bound = float("inf") if le == "+Inf" else float(le)
        if bound <= concurrency:
            decode += n
        else:
            prefill += n
    total = decode + prefill
    return {
        "decode_only_steps": decode,
        "prefill_heavy_steps": prefill,
        "prefill_step_frac": (prefill / total) if total else None,
    }


def snapshot_delta(m0, m1, elapsed, concurrency):
    d = {"elapsed_sec": elapsed}
    for m in DELTA_METRICS:
        d[m.replace("vllm:", "").replace("_total", "_delta")] = parse_metric(m1, m) - parse_metric(m0, m)
    for m in HISTOGRAM_METRICS:
        hd = histogram_delta(parse_histogram(m0, m), parse_histogram(m1, m))
        key = m.replace("vllm:", "").replace("_total", "")
        d[key + "_buckets"] = hd
        d[key + "_census"] = census_split(hd, concurrency)
    return d


def run_bench(arm, ctx, out_dir, out_file, conc=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    concurrency = conc or CTX_CONCURRENCY[ctx]
    if ARMS[arm]:
        assert_arm_is_k0(concurrency)
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


def phase_grid(arm, only_ctx, conc=None):
    ctxs = [only_ctx] if only_ctx else list(CTX_CONCURRENCY)
    for ctx in ctxs:
        c = conc or CTX_CONCURRENCY[ctx]
        # Cell = (ctx, concurrency). The KV pool couples the two -- a long ctx
        # simply cannot hold many sequences -- so concurrency is part of the
        # cell identity, never an implicit consequence of ctx.
        out_dir = RESULTS_DIR / "grid" / arm / f"ctx_{ctx}_c{c}"
        out_dir.mkdir(parents=True, exist_ok=True)
        reps = int(os.environ.get("REPS", "4"))  # 1 warmup + 3 measure
        n_warm = reps - 3
        for i in range(reps):
            is_measure = i >= n_warm
            label = "measure" if is_measure else "warmup"
            seed = i - n_warm if is_measure else i
            m0 = get_metrics()
            elapsed, rc = run_bench(arm, ctx, out_dir, f"{label}_{seed}.json", conc)
            m1 = get_metrics()
            if m0 and m1:
                (out_dir / f"snapshot_{label}_{seed}.json").write_text(
                    json.dumps(snapshot_delta(m0, m1, elapsed, c), indent=2))
            print(f"    [{arm}][ctx={ctx}][{label}_{seed}] elapsed={elapsed:.1f}s rc={rc}", flush=True)


def run_arm(arm, only_ctx, conc=None):
    tag = f"{arm}{'_nsys' if NSYS else ''}{f'_ctx{only_ctx}' if only_ctx else ''}{f'_c{conc}' if conc else ''}"
    if not any(e["arm"] == arm and e["discarded"] for e in load_pools()):
        discard_first_launch(arm)
    proc, log = launch_server(arm, tag)
    try:
        tokens, gib = parse_pool(tag)
        record_launch(arm, tag, tokens, gib, discarded=False)
        record_cudagraph_mode(tag)
        cold_start_burn()
        phase_grid(arm, only_ctx, conc)
    finally:
        # keep the server log: KV pool size, cudagraph mode, any downgrade warning
        shutil.copy(f"/tmp/server_{tag}.log", RESULTS_DIR / f"server_{tag}.log")
        kill_server(proc, log)


# Order the ctx sweep so that the primary endpoint is complete first. A rental can
# be cut short -- by credit, by a dead box -- and arm-major ordering would then
# leave a full no_spec arm with no dsd_k0 to compare it against, i.e. nothing. The
# endpoints of Tax(49400)-Tax(400) are measured first, the interior fills in after.
PAIRED_CTX_ORDER = [400, 49400, 4000, 16000, 32000]


def run_paired(ctx_order):
    """ctx-major: both arms at one ctx before moving on.

    Every completed ctx yields a usable A/B pair, so an interrupted run degrades
    to a shorter sweep instead of an unusable one. Pairing the arms close in time
    also limits drift between them.
    """
    done = []
    for ctx in ctx_order:
        if ctx not in CTX_CONCURRENCY:
            print(f"[skip] ctx={ctx} has no concurrency entry", flush=True)
            continue
        print(f"\n===== ctx={ctx} (concurrency={CTX_CONCURRENCY[ctx]}) =====", flush=True)
        for arm in ("no_spec", "dsd_k0"):
            run_arm(arm, ctx)
        done.append(ctx)
        (RESULTS_DIR / "PAIRS_DONE.txt").write_text(
            "\n".join(f"{c}\t{CTX_CONCURRENCY[c]}" for c in done) + "\n")
        print(f"===== ctx={ctx} PAIR COMPLETE (done: {done}) =====", flush=True)
    assert_pools_comparable("no_spec", "dsd_k0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=[], choices=list(ARMS) + [])
    ap.add_argument("--only-ctx", type=int, default=0)
    ap.add_argument("--paired", action="store_true",
                    help="ctx-major sweep of both arms (budget-safe ordering)")
    ap.add_argument("--ctx-order", type=str, default="",
                    help="comma-separated ctx order for --paired")
    ap.add_argument("--conc", type=int, default=0,
                    help="override concurrency for this cell (ctx and batch are "
                         "coupled by the KV pool, so the pair must be stated)")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.paired:
        order = ([int(x) for x in args.ctx_order.split(",")]
                 if args.ctx_order else PAIRED_CTX_ORDER)
        run_paired(order)
        return
    for arm in args.arms:
        run_arm(arm, args.only_ctx or None, args.conc or None)


if __name__ == "__main__":
    main()
