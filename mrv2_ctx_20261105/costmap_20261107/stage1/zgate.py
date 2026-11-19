"""Z start gate (STAGE1_R2_PREREG_DRAFT_20261111.md §4), run once before the first measured launch.

Starts the registered Z server through run_costmap's own start_server (same argv, trace hook and
readiness check as a measured launch), with every compile/JIT cache redirected under ZGATE_DIR so the
measured launches start from the cache state the registration assumed. Pass requires, in order:
  start    server ready (/v1/models) within ZGATE_START_LIMIT_S; /health 200
  log      no "Traceback" / "TypeError" in server.log (attempt 2 died on a TypeError in capture)
  capture  trace rows ev=capture for role target and drafter_prefill, each with descriptors
           (K_max = 1 has no drafter_decode capture, as K1_r0 in attempt 2; recorded, not required)
  request  one completion, 16 tokens, temperature 0, ignore_eos: HTTP 200, completion_tokens 16,
           request_success +1, spec_decode draft tokens +0, and every new scheduler row selected K = 0
  stop     process group gone, port closed, GPU memory back within 512 MiB of the pre-gate reading
  caches   default cache trees byte-for-byte unchanged (path, size, mtime), then ZGATE_DIR removed
Writes OUT/zgate.json; exit 0 pass, 3 fail (reason in zgate.json), 4 unexpected error.

  DEADLINE_EPOCH=.. MODEL_DIR=.. DRAFT_DIR=.. SMOKE=/root/smoke COSTMAP=/root/costmap \
    OUT=/root/costmap_out/zgate ZGATE_DIR=/root/zgate python zgate.py
"""

import glob
import json
import os
import shutil
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

REAL_HOME = os.path.expanduser("~")
ZGATE = Path(os.environ["ZGATE_DIR"])
START_LIMIT_S = float(os.environ.get("ZGATE_START_LIMIT_S", "900"))
GPU_SLACK_MIB = 512
PROMPT = "The capital of France is"
MAX_TOKENS = 16
os.environ.setdefault("CONFIGS", "Z")
os.environ.setdefault("PLAN", "stage1")

sys.path.insert(0, os.environ["COSTMAP"])
import costmap_cmd as C  # noqa: E402
import run_costmap as R  # noqa: E402

OUT = R.OUT
DEFAULT_CACHES = [os.path.join(REAL_HOME, p) for p in (".cache", ".triton", ".nv")] + sorted(
    glob.glob("/tmp/torchinductor_*"))
REDIRECT = {
    "HOME": ZGATE / "home",
    "XDG_CACHE_HOME": ZGATE / "cache",
    "VLLM_CACHE_ROOT": ZGATE / "cache" / "vllm",
    "FLASHINFER_WORKSPACE_BASE": ZGATE / "home",
    "TRITON_CACHE_DIR": ZGATE / "triton",
    "TORCHINDUCTOR_CACHE_DIR": ZGATE / "inductor",
    "CUDA_CACHE_PATH": ZGATE / "nv",
}


class Fail(Exception):
    pass


def fingerprint(roots):
    out = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


def trace_rows(d):
    rows = []
    for p in glob.glob(str(d / "trace" / "mrv2_trace_*.jsonl")):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def port_open():
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", R.PORT)) == 0


def metrics():
    text = R.scrape_text()
    names = ("vllm:request_success_total", "vllm:spec_decode_num_draft_tokens_total")
    found = {n: R.metric_values(text, n) for n in names}
    missing = [n for n, v in found.items() if not v]
    if missing:
        raise Fail(f"request: counters absent from /metrics: {missing}")
    return {n: sum(v) for n, v in found.items()}


def get(url, data=None, timeout=10):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError as e:
        return None, repr(e).encode()


def main(report, before_caches):
    OUT.mkdir(parents=True, exist_ok=True)
    report.update(default_caches=DEFAULT_CACHES, redirect={k: str(v) for k, v in REDIRECT.items()})
    before_caches.update(fingerprint(DEFAULT_CACHES))
    report["gpu_before"] = gpu0 = R.gpu_apps()
    if ZGATE.exists():
        raise Fail(f"setup: {ZGATE} already exists")
    for v in REDIRECT.values():
        v.mkdir(parents=True, exist_ok=True)
    os.environ.update({k: str(v) for k, v in REDIRECT.items()})
    R.DEADLINE = min(R.DEADLINE, time.time() + START_LIMIT_S)
    proc = None
    try:
        proc, startup = R.start_server(OUT, C.spec("Z", R.DRAFT))
        report["startup_s"] = round(startup)
        if proc is None:
            raise Fail("start: server not ready (see server.log)")
        health, _ = get(f"http://127.0.0.1:{R.PORT}/health")
        report["health"] = health
        if health != 200:
            raise Fail(f"start: /health {health}")

        log = (OUT / "server.log").read_text(errors="replace")
        bad = [w for w in ("Traceback", "TypeError") if w in log]
        if bad:
            raise Fail(f"log: {bad} in server.log")

        time.sleep(R.FLUSH_WAIT_S)
        rows = trace_rows(OUT)
        caps = {}
        for r in rows:
            if r.get("ev") == "capture":
                caps.setdefault(r.get("role"), []).append(len(r.get("descs") or []))
        report["captures"] = caps
        missing = [role for role in ("target", "drafter_prefill") if not any(caps.get(role, []))]
        if missing:
            raise Fail(f"capture: no capture row with descriptors for {missing}")

        n_sched = sum(r.get("ev") == "sched" for r in rows)
        m0 = metrics()
        body = json.dumps({"model": R.MODEL, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0,
                           "ignore_eos": True}).encode()
        status, raw = get(f"http://127.0.0.1:{R.PORT}/v1/completions", data=body, timeout=300)
        try:
            out = json.loads(raw)
        except ValueError:
            out = {}
        time.sleep(R.FLUSH_WAIT_S)
        m1 = metrics()
        new_sched = [r for r in trace_rows(OUT) if r.get("ev") == "sched"][n_sched:]
        report["request"] = {
            "status": status, "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
            "success_delta": m1["vllm:request_success_total"] - m0["vllm:request_success_total"],
            "draft_tokens_delta": m1["vllm:spec_decode_num_draft_tokens_total"]
            - m0["vllm:spec_decode_num_draft_tokens_total"],
            "sched_rows": len(new_sched), "selected_k": sorted({r.get("k") for r in new_sched}),
        }
        q = report["request"]
        if q["status"] != 200 or q["completion_tokens"] != MAX_TOKENS:
            raise Fail(f"request: status {q['status']} completion_tokens {q['completion_tokens']}")
        if q["success_delta"] != 1:
            raise Fail(f"request: request_success delta {q['success_delta']} (not this request's server?)")
        if q["draft_tokens_delta"] != 0:
            raise Fail(f"request: draft tokens delta {q['draft_tokens_delta']} at K=0")
        if not new_sched or q["selected_k"] != [0]:
            raise Fail(f"request: scheduler rows {q['sched_rows']} with selected K {q['selected_k']}")
    finally:
        report["group_gone"] = R.kill_group(proc) if proc is not None else None
    if not report["group_gone"]:
        raise Fail("stop: server process group not gone")
    if port_open():
        raise Fail(f"stop: port {R.PORT} still open")
    for _ in range(30):
        report["gpu_after"] = gpu1 = R.gpu_apps()
        if (gpu0["memory_used_mib"] is None or (gpu1["memory_used_mib"] or 0) <= gpu0["memory_used_mib"] + GPU_SLACK_MIB):
            break
        time.sleep(2)
    else:
        raise Fail(f"stop: GPU memory {gpu1['memory_used_mib']} MiB vs {gpu0['memory_used_mib']} before")


def finish(report, before_caches, reason):
    after = fingerprint(DEFAULT_CACHES)
    changed = sorted(p for p in set(before_caches) | set(after) if before_caches.get(p) != after.get(p))
    report["default_cache_files"] = len(before_caches)
    report["default_cache_changed"] = changed[:50]
    if changed and reason is None:
        reason = f"caches: {len(changed)} default cache files changed"
    shutil.rmtree(ZGATE, ignore_errors=True)
    report["zgate_dir_removed"] = not ZGATE.exists()
    if not report["zgate_dir_removed"] and reason is None:
        reason = f"caches: {ZGATE} not removed"
    report["pass"] = reason is None
    report["reason"] = reason
    R.write_json(OUT / "zgate.json", report)
    R.log(f"ZGATE pass={report['pass']} {reason or ''}")
    return 0 if reason is None else 3


if __name__ == "__main__":
    report, before = {}, {}
    try:
        main(report, before)
        reason, code = None, 0
    except Fail as e:
        reason, code = str(e), 3
    except BaseException:
        reason, code = "unexpected: " + traceback.format_exc()[-1500:], 4
    sys.exit(finish(report, before, reason) and (code or 3))
