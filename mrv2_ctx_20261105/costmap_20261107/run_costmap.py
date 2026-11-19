"""Cost map runner on the box (DESIGN.md §2-5, PILOT.md §2). One launch per configuration in
CONFIGS, cells in the fixed plan order, then OUT/ALL_DONE; any stop writes OUT/STOPPED with the
reason and nothing after it starts.

  CONFIGS=K3  DEADLINE_EPOCH=.. MODEL_DIR=.. DRAFT_DIR=.. SMOKE=/root/smoke \
    COSTMAP=/root/costmap OUT=/root/costmap_out python run_costmap.py
Exit: 0 done, 2 blocked (deadline / server), 3 gate failed, 4 unexpected error.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, os.environ["COSTMAP"])
import analyze_costmap as A  # noqa: E402
import costmap_cmd as C  # noqa: E402

MODEL, DRAFT = os.environ["MODEL_DIR"], os.environ["DRAFT_DIR"]
SMOKE, OUT = Path(os.environ["SMOKE"]), Path(os.environ["OUT"])
DEADLINE = float(os.environ["DEADLINE_EPOCH"])
CONFIGS = os.environ["CONFIGS"].split(",")
PLAN_SEED = int(os.environ.get("PLAN_SEED", "20261107"))
PLAN = os.environ.get("PLAN", "pilot")
PORT = int(os.environ.get("PORT", "8000"))
CELL_TIMEOUT_S = float(os.environ.get("CELL_TIMEOUT_S", "900"))
FLUSH_WAIT_S = 2.5
EST = {"launch": float(os.environ.get("LAUNCH_EST_S", 30 * 60))}
COUNTERS = ("vllm:num_preemptions_total", "vllm:request_success_total",
            "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total")
LOG = OUT / "run.log"


class Stop(Exception):
    def __init__(self, reason, code):
        super().__init__(reason)
        self.reason, self.code = reason, code


def log(msg):
    line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def metric_values(text, name):
    return [float(m) for m in re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})? (\S+)$", text, re.M)]


def scrape_text():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    except OSError:
        return ""


def scrape():
    text = scrape_text()
    return {name: sum(metric_values(text, name)) for name in COUNTERS} if text else {}


class Sampler(threading.Thread):
    """KV usage and running requests once per second while the server is up."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.f, self.stop_flag = open(path, "a"), threading.Event()

    def run(self):
        while not self.stop_flag.wait(1.0):
            text = scrape_text()
            if not text:
                continue
            kv, run = metric_values(text, "vllm:kv_cache_usage_perc"), metric_values(text, "vllm:num_requests_running")
            self.f.write(json.dumps({"t_ns": time.monotonic_ns(), "kv": max(kv) if kv else None,
                                     "running": sum(run) if run else None}) + "\n")
            self.f.flush()

    def close(self):
        self.stop_flag.set()
        self.join(timeout=5)
        self.f.close()


def group_alive(pgid):
    """Live (non-zombie) members of a process group."""
    r = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    live = []
    for pid in r.stdout.split():
        st = subprocess.run(["ps", "-o", "stat=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if st and not st.startswith("Z"):
            live.append(int(pid))
    return live


def kill_group(proc, timeout=120):
    """SIGTERM the process group started with `proc` (its pgid is proc.pid, start_new_session),
    then SIGKILL, even when the leader has already exited; True when no live member is left."""
    if proc is None:
        return True
    pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        t0 = time.time()
        while time.time() - t0 < wait:
            proc.poll()
            if not group_alive(pgid):
                return True
            time.sleep(0.5)
    proc.poll()
    return not group_alive(pgid)


def start_server(d, s):
    d.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "MRV2_TRACE_PER_REQUEST"}
    env.update(MRV2_TRACE_DIR=str(d / "trace"), PYTHONPATH=str(SMOKE / "trace"), VLLM_USE_V2_MODEL_RUNNER="1")
    cmd = C.serve_argv(MODEL, s, PORT)
    (d / "cmd.json").write_text(json.dumps(cmd))
    proc = subprocess.Popen(cmd, env=env, stdout=open(d / "server.log", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return proc, time.time() - t0
        except OSError:
            if proc.poll() is not None or time.time() > DEADLINE:
                kill_group(proc)
                return None, time.time() - t0
            time.sleep(5)


def gpu_apps():
    """Compute processes (may be hidden inside a container) and total GPU memory in use."""
    def q(args):
        try:
            return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""
    apps = []
    for line in q(["--query-compute-apps=pid,used_memory"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            apps.append({"pid": parts[0], "used_mib": int(parts[1])})
    used = [int(x) for x in q(["--query-gpu=memory.used"]).split() if x.strip().isdigit()]
    return {"apps": apps, "memory_used_mib": max(used) if used else None}


def gpu_logger(d):
    q = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    return subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv", "-l", "1"],
                            stdout=open(d / "nvidia_smi.csv", "w"), stderr=subprocess.STDOUT, start_new_session=True)


def bench(d, cell):
    cmd = C.bench_argv(MODEL, PORT, cell, d)
    before = scrape()
    t0 = time.monotonic_ns()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    timeout = False
    try:
        so, se = proc.communicate(timeout=min(CELL_TIMEOUT_S, max(60.0, DEADLINE - time.time())))
    except subprocess.TimeoutExpired:
        kill_group(proc, timeout=10)
        so, se = proc.communicate()
        timeout = True
    t1 = time.monotonic_ns()
    after = scrape()
    (d / f"{cell['id']}.log").write_text((so or "")[-20000:] + "\n--- stderr ---\n" + (se or "")[-8000:])
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after} if before and after else {}
    tpot = None
    try:
        tpot = json.load(open(d / f"{cell['id']}.json")).get("median_tpot_ms")
    except (OSError, ValueError):
        pass
    return dict(cell, rc=proc.returncode, timeout=timeout, t0_ns=t0, t1_ns=t1, metrics_delta=delta,
                bench_tpot_ms=tpot, preempted=delta.get("vllm:num_preemptions_total", 0) > 0)


def analyze_now(name, config, rep, status):
    L = {"config": config, "k": A.CONFIG_K[config], "rep": rep, "dir": name, "status": status,
         "plan_seed": PLAN_SEED, "plan": PLAN}
    return A.load_launch(str(OUT), L)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=str))


def launch(config, rep):
    if time.time() + EST["launch"] * 1.2 > DEADLINE:
        raise Stop(f"BLOCKED_DEADLINE before {config} rep {rep} (estimate {EST['launch'] * 1.2:.0f}s)", 2)
    name = f"{config}_r{rep}"
    d = OUT / name
    t_start = time.time()
    proc, startup = start_server(d, C.spec(config, DRAFT))
    entry = {"config": config, "k": A.CONFIG_K[config], "rep": rep, "dir": name, "plan_seed": PLAN_SEED, "plan": PLAN,
             "startup_s": round(startup)}
    if proc is None:
        entry["status"] = "stopped:server_not_ready"
        with open(OUT / "launches.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        raise Stop(f"BLOCKED_SERVER {name} (see {name}/server.log)", 2)
    log(f"{name} startup_s={startup:.0f}")
    apps = gpu_apps()
    write_json(d / "gpu_apps.json", apps)
    smi, sampler = gpu_logger(d), Sampler(d / "metrics.jsonl")
    sampler.start()
    skip_above, measured, stop, status = {}, 0, None, "complete"
    try:
        with open(d / "cells.jsonl", "w") as f:
            for cell in A.plan(PLAN_SEED, PLAN):
                if cell["bs"] > skip_above.get(cell["ctx"], 10**9):
                    f.write(json.dumps(dict(cell, skipped="larger BS than an infeasible cell")) + "\n")
                    continue
                if time.time() > DEADLINE - 120:
                    stop = Stop(f"DEADLINE reached in {name} before {cell['id']}", 2)
                    break
                r = bench(d, cell)
                f.write(json.dumps(r) + "\n")
                f.flush()
                log(f"{name} {cell['id']} rc={r['rc']} timeout={r['timeout']} "
                    f"s={(r['t1_ns'] - r['t0_ns']) / 1e9:.0f} preempt={r['preempted']} tpot={r['bench_tpot_ms']}")
                if r["preempted"]:
                    skip_above[cell["ctx"]] = min(skip_above.get(cell["ctx"], 10**9), cell["bs"])
                if proc.poll() is not None:
                    stop = Stop(f"SERVER_EXITED in {name} during {cell['id']} (rc {proc.returncode})", 2)
                    break
                if r["timeout"]:
                    stop = Stop(f"CELL_TIMEOUT {name} {cell['id']}", 2)
                    break
                if cell["kind"] != "warmup":
                    measured += 1
                if measured == A.EARLY_CELLS:
                    f.flush()
                    time.sleep(FLUSH_WAIT_S)
                    g = A.early_gate(analyze_now(name, config, rep, "running")["cells"])
                    write_json(d / "early_gate.json", g)
                    log(f"{name} early_gate pass={g['pass']} {g['reasons']}")
                    if not g["pass"]:
                        stop = Stop(f"EARLY_GATE_FAILED {name}: {'; '.join(g['reasons'])}", 3)
                        break
        time.sleep(FLUSH_WAIT_S)
    except Stop as e:
        stop = e
    finally:
        gone = False
        for what, fn in (("sampler", sampler.close), ("server", lambda: kill_group(proc)),
                         ("gpu_logger", lambda: kill_group(smi, timeout=10))):
            try:
                ok = fn()
                gone = ok if what == "server" else gone
            except Exception as e:
                log(f"{name} cleanup {what} failed: {e!r}")
        log(f"{name} server stopped, process group gone={gone}")
    if stop is None and not gone:
        stop = Stop(f"SERVER_NOT_REAPED {name}", 2)
    sg = subprocess.run([sys.executable, str(SMOKE / "analyze_smoke.py"), str(d / "trace")],
                        capture_output=True, text=True)
    (d / "gates.json").write_text(sg.stdout)
    try:
        smoke_gates = json.loads(sg.stdout)
    except ValueError:
        smoke_gates = {}
    status = "complete" if stop is None else "stopped:" + stop.reason.split()[0]
    L = analyze_now(name, config, rep, status)
    gate = A.launch_gate(L, smoke_gates, (d / "server.log").read_text(errors="replace"), apps)
    write_json(d / "launch_gate.json", gate)
    write_json(d / "launch_report.json", L)
    log(f"{name} launch_gate pass={gate['pass']} valid_share={gate['valid_share']} {gate['reasons']} "
        f"drift={L['sentinel_drift_log']}")
    if stop is None and not gate["pass"]:
        stop = Stop(f"LAUNCH_GATE_FAILED {name}: {'; '.join(gate['reasons'])}", 3)
        status = "stopped:LAUNCH_GATE_FAILED"
    entry.update(status=status, wall_s=round(time.time() - t_start))
    with open(OUT / "launches.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    EST["launch"] = max(EST["launch"], time.time() - t_start)
    if stop is not None:
        raise stop


def on_sigterm(signum, frame):
    raise Stop(f"RUNNER_SIGNAL {signum}", 2)


def main():
    signal.signal(signal.SIGTERM, on_sigterm)
    OUT.mkdir(parents=True, exist_ok=True)
    for marker in ("ALL_DONE", "STOPPED"):
        if (OUT / marker).exists():
            print(f"{marker} already present in {OUT}; refusing to continue", file=sys.stderr)
            sys.exit(2)
    log(f"start configs={CONFIGS} plan={PLAN} plan_seed={PLAN_SEED} deadline={time.strftime('%H:%M:%SZ', time.gmtime(DEADLINE))}")
    write_json(OUT / "plan.json", A.plan(PLAN_SEED, PLAN))
    done = {}
    if (OUT / "launches.jsonl").exists():
        for L in map(json.loads, open(OUT / "launches.jsonl")):
            done[L["config"]] = max(done.get(L["config"], -1), L["rep"])
    try:
        for config in CONFIGS:
            rep = done.get(config, -1) + 1
            done[config] = rep
            launch(config, rep)
    except Stop as e:
        (OUT / "STOPPED").write_text(e.reason + "\n")
        log(f"STOPPED {e.reason}")
        sys.exit(e.code)
    except BaseException:
        (OUT / "STOPPED").write_text("UNEXPECTED\n" + traceback.format_exc())
        log("STOPPED UNEXPECTED")
        sys.exit(4)
    (OUT / "ALL_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
