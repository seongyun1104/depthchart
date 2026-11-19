"""Per-process trace of dynamic-SD K decisions and MRv2 CUDA graph use.

Active only when MRV2_TRACE_DIR is set; writes mrv2_trace_<pid>.jsonl there.

Scheduler rows (`ev=sched`): batch size, the context length the K lookup read
(the scheduler module's `dynamic_sd_context_length`, wrapped), the selected K,
the drafts verified on this step and `skip_drafts`. `ev=drafts`: drafts the
worker handed back after the step. With MRV2_TRACE_PER_REQUEST=1, `ev=accept`
per request and step: drafts verified (padding drafts excluded), tokens sampled
(accepted = sampled - 1), and whether the request finished on that step (a
finish can cut the count, so such rows are censored). Drafts never verified
are not rows and are not counted as rejections.

Graph rows, per manager role (target / drafter_prefill / drafter_decode):
`ev=capture` (captured descriptors, seconds, free-memory drop when CUDA is
present), `ev=dispatch` (inputs and the chosen descriptor), `ev=replay`
(a FULL graph actually replayed via run_fullgraph), `ev=pw` (run_pw_graph),
`ev=specialize` (drafter decode descriptor after K specialisation).
Only the outermost call of a wrapped method is recorded.

Engine rows. `ev=wait`: time spent in the `_wait_for_output` that ran for one step
(`wait_ns`, written when it returns), with the instance class (`cls`) and the class whose
method ran (`impl`). Every class of the engine module that defines its own
`_wait_for_output` is wrapped (EngineCore, EngineCoreProc on 7ecc83a); only the outermost
call is recorded, so an override that calls super() is counted once. In EngineCoreProc the
wait polls the output and meanwhile handles client requests from the input queue, so
`wait_ns` is the time until the step's result was taken, including that request handling;
it is not GPU time. `ev=input`: one row per client request handled by EngineCoreProc
(`type`, `in_wait`), so request arrivals during a measured window are visible. With async
scheduling the busy loop schedules a step, submits it, then waits for the oldest queued
result, so the interval between `sched` rows is the loop period. The file is flushed at
least once per second, so rows survive a server that is killed without running atexit.
"""

import atexit
import functools
import json
import os
import threading
import time
from collections import Counter

from post_import import on_import

SCHED = "vllm.v1.core.sched.scheduler"
GRAPHS = "vllm.v1.worker.gpu.cudagraph_utils"
ENGINE = "vllm.v1.engine.core"
_local = threading.local()
_tracer = None


class Tracer:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1 << 16)
        self.n = 0
        atexit.register(self.close)
        threading.Thread(target=self._flush_loop, daemon=True).start()

    def _flush_loop(self):
        while not self.f.closed:
            time.sleep(1.0)
            try:
                self.f.flush()
            except ValueError:
                return

    def write(self, **row):
        row["t_ns"] = time.monotonic_ns()
        self.f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        self.n += 1
        if self.n % 256 == 0:
            self.f.flush()

    def close(self):
        if not self.f.closed:
            self.f.flush()
            self.f.close()


def tracer():
    global _tracer
    if _tracer is None:
        out = os.environ["MRV2_TRACE_DIR"]
        os.makedirs(out, exist_ok=True)
        _tracer = Tracer(os.path.join(out, f"mrv2_trace_{os.getpid()}.jsonl"))
    return _tracer


def desc_row(d):
    row = {
        "mode": str(d.cg_mode).split(".")[-1],
        "tokens": d.num_tokens,
        "reqs": d.num_reqs,
        "q": d.uniform_token_count,
        "max_q": d.max_query_len,
        "loras": d.num_active_loras,
        "ub": d.num_ubatches,
    }
    k = getattr(d, "num_speculative_tokens", None)
    if k is not None:
        row["k"] = k
    return row


def outermost(name):
    def wrap(fn):
        @functools.wraps(fn)
        def inner(self, *a, **k):
            depth = getattr(_local, name, 0)
            setattr(_local, name, depth + 1)
            try:
                return fn(self, depth == 0, *a, **k)
            finally:
                setattr(_local, name, depth)

        return inner

    return wrap


def role(mgr):
    cls = type(mgr).__name__
    if cls == "ModelCudaGraphManager":
        return "target"
    flag = getattr(mgr, "draft_decode", None)
    if flag is not None:
        return "drafter_decode" if flag else "drafter_prefill"
    lens = getattr(mgr, "decode_max_reqs", {}) or {}
    if lens and max(lens) == 1:
        return "drafter_decode"
    return "drafter_prefill" if "Speculator" in cls else cls


def patch_class(cls):
    if cls.__dict__.get("_mrv2_traced"):
        return
    for name in ("capture", "run_fullgraph", "run_pw_graph", "specialize_spec_tokens"):
        fn = cls.__dict__.get(name)
        if fn is not None:
            setattr(cls, name, WRAPPERS[name](fn))
    cls._mrv2_traced = True


def _capture(fn):
    @outermost("capture")
    def inner(self, top, *a, **k):
        if not top:
            return fn(self, *a, **k)
        free0 = _free_bytes()
        t0 = time.monotonic()
        out = fn(self, *a, **k)
        tracer().write(
            ev="capture",
            role=role(self),
            seconds=round(time.monotonic() - t0, 3),
            free_drop_bytes=None if free0 is None else free0 - _free_bytes(),
            graphs=Counter(str(d.cg_mode).split(".")[-1] for d in self.graphs),
            descs=[desc_row(d) for d in self.graphs],
        )
        return out

    return inner


def _replay(fn):
    @outermost("replay")
    def inner(self, top, desc, *a, **k):
        if top:
            tracer().write(ev="replay", role=role(self), desc=desc_row(desc))
        return fn(self, desc, *a, **k)

    return inner


def _pw(fn):
    @outermost("pw")
    def inner(self, top, *a, **k):
        if top:
            tracer().write(ev="pw", role=role(self))
        return fn(self, *a, **k)

    return inner


def _specialize(fn):
    @outermost("specialize")
    def inner(self, top, desc, k_steps, *a, **k):
        out = fn(self, desc, k_steps, *a, **k)
        if top:
            tracer().write(ev="specialize", role=role(self), k=k_steps, desc=desc_row(out))
        return out

    return inner


WRAPPERS = {
    "capture": _capture,
    "run_fullgraph": _replay,
    "run_pw_graph": _pw,
    "specialize_spec_tokens": _specialize,
}


def _free_bytes():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0]
    except Exception:
        pass
    return None


def install_graphs(module):
    base = module.CudaGraphManager
    if base.__dict__.get("_mrv2_init_traced"):
        return
    init, dispatch = base.__init__, base.dispatch

    def traced_init(self, *a, **k):
        init(self, *a, **k)
        for cls in type(self).__mro__:
            if cls is not object and issubclass(cls, base):
                patch_class(cls)

    def traced_dispatch(self, num_reqs, num_tokens, uniform_token_count, *a, **k):
        out = dispatch(self, num_reqs, num_tokens, uniform_token_count, *a, **k)
        if self._graphs_captured:
            tracer().write(
                ev="dispatch",
                role=role(self),
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                q=uniform_token_count,
                desc=desc_row(out),
            )
        return out

    base.__init__ = traced_init
    base.dispatch = traced_dispatch
    base._mrv2_init_traced = True


def install_scheduler(module):
    Scheduler = module.Scheduler
    if Scheduler.__dict__.get("_mrv2_traced"):
        return
    ctx_fn = module.dynamic_sd_context_length

    def traced_ctx(requests):
        value = ctx_fn(requests)
        _local.ctx = value
        return value

    module.dynamic_sd_context_length = traced_ctx
    schedule = Scheduler.schedule

    def traced_schedule(self, *a, **k):
        _local.ctx = None
        before = {
            r: (q.num_computed_tokens, q.num_prompt_tokens)
            for r, q in self.requests.items()
        }
        out = schedule(self, *a, **k)
        if out.num_scheduled_tokens:
            ids = list(out.num_scheduled_tokens)
            snap = [before.get(r, (0, 0)) for r in ids]
            pool = [c for c, p in snap if c >= p] or [c for c, _ in snap]
            verified = Counter(
                len(out.scheduled_spec_decode_tokens.get(r, ())) for r in ids
            )
            tracer().write(
                ev="sched",
                bs=len(ids),
                prefill_rows=sum(1 for c, p in snap if c < p),
                tokens=out.total_num_scheduled_tokens,
                ctx=_local.ctx,
                ctx_snapshot=sum(pool) // len(pool),
                k=out.num_spec_tokens_to_schedule,
                skip_drafts=out.skip_drafts,
                verified=dict(verified),
            )
        return out

    def traced_update(fn):
        def inner(self, draft_token_ids, *a, **k):
            tracer().write(
                ev="drafts",
                via=fn.__name__,
                produced=dict(
                    Counter(len(d) for d in draft_token_ids.draft_token_ids)
                ),
            )
            return fn(self, draft_token_ids, *a, **k)

        return inner

    if os.environ.get("MRV2_TRACE_PER_REQUEST"):
        update_from_output = Scheduler.update_from_output

        def traced_update_from_output(self, scheduler_output, runner_output):
            spec = scheduler_output.scheduled_spec_decode_tokens
            invalid = scheduler_output.num_invalid_spec_tokens or {}
            index = runner_output.req_id_to_index
            sampled = runner_output.sampled_token_ids
            out = update_from_output(self, scheduler_output, runner_output)
            for req_id, drafts in spec.items():
                i = index.get(req_id)
                if i is None or i >= len(sampled):
                    continue
                req = self.requests.get(req_id)
                tracer().write(
                    ev="accept",
                    req=req_id,
                    verified=len(drafts) - invalid.get(req_id, 0),
                    sampled=len(sampled[i]),
                    finished=req is None or req.is_finished(),
                )
            return out

        Scheduler.update_from_output = traced_update_from_output
    Scheduler.schedule = traced_schedule
    for name in ("update_draft_token_ids", "update_draft_token_ids_in_output"):
        setattr(Scheduler, name, traced_update(getattr(Scheduler, name)))
    Scheduler._mrv2_traced = True


def _traced_wait(fn, impl):
    @functools.wraps(fn)
    def inner(self, future):
        if getattr(_local, "in_wait", False):
            return fn(self, future)
        _local.in_wait = True
        t0 = time.monotonic_ns()
        try:
            return fn(self, future)
        finally:
            _local.in_wait = False
            tracer().write(ev="wait", wait_ns=time.monotonic_ns() - t0,
                           cls=type(self).__name__, impl=impl)

    inner._mrv2_traced = True
    return inner


def _traced_input(fn):
    @functools.wraps(fn)
    def inner(self, request_type, request):
        tracer().write(ev="input", type=getattr(request_type, "name", str(request_type)),
                       in_wait=getattr(_local, "in_wait", False))
        return fn(self, request_type, request)

    inner._mrv2_traced = True
    return inner


def install_engine(module):
    for cls in vars(module).values():
        if not isinstance(cls, type) or cls.__module__ != module.__name__:
            continue
        fn = cls.__dict__.get("_wait_for_output")
        if fn is not None and not getattr(fn, "_mrv2_traced", False):
            cls._wait_for_output = _traced_wait(fn, cls.__name__)
        fn = cls.__dict__.get("_handle_client_request")
        if fn is not None and not getattr(fn, "_mrv2_traced", False):
            cls._handle_client_request = _traced_input(fn)


def activate():
    if os.environ.get("MRV2_TRACE_DIR"):
        on_import(SCHED, install_scheduler)
        on_import(GRAPHS, install_graphs)
        on_import(ENGINE, install_engine)
