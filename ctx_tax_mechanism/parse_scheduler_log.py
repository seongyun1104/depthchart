#!/usr/bin/env python3
"""
Scheduler-step census from a DEBUG vLLM server log.

Answers the chunked-prefill-vs-decode attribution (Suppressor72's lead: the ctx
tax concentrates in chunked prefill, not steady decode). For each scheduler step
we recover scheduled tokens and running/waiting counts, classify the step as
prefill-heavy vs decode-only, and summarize per arm.

!! The DEBUG line format is vLLM-version-specific. Confirm the regexes against a
real server_*.log on the rental before trusting the classification; unmatched
lines are counted and reported so a format drift is visible, not silent.
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

# Candidate patterns for V1 scheduler/engine debug lines. Extend on the rental
# after inspecting the actual log. Each returns a dict of ints when it matches.
PATTERNS = [
    re.compile(r"num_scheduled_tokens[=:]\s*(?P<sched>\d+).*?"
               r"(?:num_running_reqs|running)[=:]\s*(?P<running>\d+)", re.I),
    re.compile(r"scheduled\s+(?P<sched>\d+)\s+tokens.*?(?P<running>\d+)\s+running", re.I),
    re.compile(r"Step.*?tokens[=:]\s*(?P<sched>\d+).*?waiting[=:]\s*(?P<waiting>\d+)", re.I),
]


def parse(path: Path):
    steps = []
    unmatched_stat_like = 0
    for line in path.read_text(errors="ignore").split("\n"):
        if "token" not in line.lower():
            continue
        hit = None
        for pat in PATTERNS:
            m = pat.search(line)
            if m:
                hit = {k: int(v) for k, v in m.groupdict().items() if v is not None}
                break
        if hit:
            steps.append(hit)
        elif re.search(r"\bschedul", line, re.I):
            unmatched_stat_like += 1
    return steps, unmatched_stat_like


def summarize(steps):
    # A step is "prefill-heavy" when scheduled tokens far exceed the running
    # request count (i.e. multi-token prefill chunks dominate); "decode-only"
    # when scheduled tokens ~= running requests (~1 token/req).
    prefill, decode, ambiguous = 0, 0, 0
    prefill_tokens, decode_tokens = 0, 0
    for s in steps:
        sched = s.get("sched")
        running = s.get("running")
        if sched is None:
            ambiguous += 1
            continue
        if running and sched > 2 * running:
            prefill += 1
            prefill_tokens += sched
        elif running:
            decode += 1
            decode_tokens += sched
        else:
            ambiguous += 1
    return {
        "steps_total": len(steps),
        "prefill_steps": prefill,
        "decode_steps": decode,
        "ambiguous_steps": ambiguous,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="server_*.log DEBUG files")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    report = {}
    for lp in args.logs:
        p = Path(lp)
        steps, unmatched = parse(p)
        s = summarize(steps)
        s["unmatched_scheduler_lines"] = unmatched
        if unmatched > 0 and s["steps_total"] == 0:
            s["WARNING"] = "0 steps parsed but scheduler lines present — regexes need updating for this build"
        report[p.name] = s
        print(f"{p.name}: {json.dumps(s)}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
