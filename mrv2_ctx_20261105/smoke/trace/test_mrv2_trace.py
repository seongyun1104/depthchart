"""CPU checks of mrv2_trace against the real Scheduler and CudaGraphManager of the
dc/bs-ctx-dsd tree. Run from the vLLM tree root:

  MRV2_TRACE_DIR=<tmp> VLLM_TARGET_DEVICE=cpu \
    PYTHONPATH=.:<this dir> python -m pytest -q <this dir>/test_mrv2_trace.py
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import mrv2_trace
import torch

os.environ.setdefault("MRV2_TRACE_PER_REQUEST", "1")

mrv2_trace.activate()

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from tests.v1.cudagraph.test_cudagraph_manager import (  # noqa: E402
    _create_decode_vllm_config,
)
from vllm.config import CUDAGraphMode, VllmConfig  # noqa: E402
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils as cgu  # noqa: E402

CTX = [(1, 8, 1, 64, 2), (1, 8, 65, 4096, 4)]


def rows(ev):
    mrv2_trace.tracer().f.flush()
    path = os.path.join(
        os.environ["MRV2_TRACE_DIR"], f"mrv2_trace_{os.getpid()}.jsonl"
    )
    return [r for r in map(json.loads, open(path)) if r["ev"] == ev]


def test_scheduler_rows_separate_selected_produced_and_verified():
    sched = create_scheduler(
        max_num_seqs=8,
        num_speculative_tokens=4,
        num_speculative_tokens_per_batch_size_and_context=CTX,
    )
    reqs = create_requests(num_requests=2, num_tokens=100)
    for r in reqs:
        sched.add_request(r)
    ids = [r.request_id for r in reqs]
    n0 = len(rows("sched"))
    for step in range(2):
        out = sched.schedule()
        sched.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=[[0], [0]] if step == 0 else [[0, 0], [0]],
            ),
        )
        sched.update_draft_token_ids(
            DraftTokenIds(ids, [[1] * out.num_spec_tokens_to_schedule] * 2)
        )
    sched_rows = rows("sched")[n0:]
    assert [r["bs"] for r in sched_rows] == [2, 2]
    # Prefill step: no decode rows, the lookup reads all rows (computed 0).
    assert sched_rows[0]["ctx"] == 0 and sched_rows[0]["k"] == 2
    assert sched_rows[0]["verified"] == {"0": 2}
    # Decode step: ctx 100 > 64 -> K 4; it verifies the 2 drafts of step 0.
    assert sched_rows[1]["ctx"] == 100 and sched_rows[1]["ctx_snapshot"] == 100
    assert sched_rows[1]["k"] == 4
    assert sched_rows[1]["verified"] == {"2": 2}
    assert rows("drafts")[-1]["produced"] == {"4": 2}
    accept = rows("accept")[-2:]
    assert [(a["verified"], a["sampled"]) for a in accept] == [(2, 2), (2, 1)]


def test_graph_rows_record_dispatch_and_actual_replay(monkeypatch):
    monkeypatch.setattr(
        cgu,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(cgu.current_platform, "get_global_graph_pool", lambda: object())
    monkeypatch.setattr(cgu, "get_offloader", MagicMock)
    config = _create_decode_vllm_config([1, 2, 4, 8], num_speculative_tokens=1)
    mgr = cgu.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_lens=VllmConfig.get_decode_cudagraph_query_lens(config, 2),
    )
    for descs in mgr._capture_descs.values():
        for d in descs:
            mgr.graphs[d] = MagicMock()
    mgr._graphs_captured = True
    desc = mgr.dispatch(num_reqs=3, num_tokens=6, uniform_token_count=2, num_active_loras=0)
    assert desc.cg_mode == CUDAGraphMode.FULL
    mgr.run_fullgraph(desc)
    mgr.graphs[desc].replay.assert_called_once()
    d = rows("dispatch")[-1]
    assert (d["num_reqs"], d["num_tokens"], d["q"]) == (3, 6, 2)
    assert d["desc"]["mode"] == "FULL" and d["desc"]["reqs"] == 4
    assert rows("replay")[-1]["desc"] == d["desc"]


def test_role_follows_declared_draft_decode(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )

    monkeypatch.setattr(
        cgu,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(cgu.current_platform, "get_global_graph_pool", lambda: object())
    monkeypatch.setattr(cgu, "get_offloader", MagicMock)
    config = _create_decode_vllm_config([1, 2, 4, 8], num_speculative_tokens=1)
    # Query length 1 for both, as Z's prefill manager has: only the flag tells them apart.
    kwargs = dict(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_lens={1: 8},
    )
    decode = SpeculatorCudaGraphManager(**kwargs, draft_decode=True)
    prefill = SpeculatorCudaGraphManager(**kwargs)
    target = object.__new__(cgu.ModelCudaGraphManager)
    assert mrv2_trace.role(decode) == "drafter_decode"
    assert mrv2_trace.role(prefill) == "drafter_prefill"
    assert mrv2_trace.role(target) == "target"


class _PolledFuture:
    """Fixture: a done future whose poll() reports not-ready once, so the
    EngineCoreProc override enters its polling loop."""

    def __init__(self, value):
        self.value, self.polls = value, 0

    def poll(self):
        self.polls += 1
        return self.polls > 1

    def result(self):
        return self.value


def test_wait_rows_for_engine_core_proc_override():
    import queue

    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCoreProc

    # Fixture instance: only the attributes the production override reads.
    proc = object.__new__(EngineCoreProc)
    proc.input_queue = queue.Queue()
    proc._deferred_inputs = []
    proc.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
    n0 = len(rows("wait"))
    i0 = len(rows("input"))
    assert proc._wait_for_output(_PolledFuture("out")) == "out"
    (w,) = rows("wait")[n0:]
    assert (w["cls"], w["impl"]) == ("EngineCoreProc", "EngineCoreProc") and w["wait_ns"] >= 0
    (inp,) = rows("input")[i0:]
    assert inp == {**inp, "type": "WAKEUP", "in_wait": True}


def test_wait_wrapper_counts_super_once_and_survives_reinstall():
    import types

    import pytest

    mod = types.ModuleType("fake_engine_core")

    class Base:
        def _wait_for_output(self, future):
            if future == "raise":
                raise ValueError("boom")
            return ("base", future)

    class Sub(Base):
        def _wait_for_output(self, future):
            return ("sub",) + super()._wait_for_output(future)

    for c in (Base, Sub):
        c.__module__ = mod.__name__
        setattr(mod, c.__name__, c)
    mrv2_trace.install_engine(mod)
    first = Sub.__dict__["_wait_for_output"]
    mrv2_trace.install_engine(mod)
    assert Sub.__dict__["_wait_for_output"] is first
    n0 = len(rows("wait"))
    assert Sub()._wait_for_output(7) == ("sub", "base", 7)
    assert [(r["cls"], r["impl"]) for r in rows("wait")[n0:]] == [("Sub", "Sub")]
    assert Base()._wait_for_output(8) == ("base", 8)
    with pytest.raises(ValueError, match="boom"):
        Sub()._wait_for_output("raise")
    assert [r["impl"] for r in rows("wait")[n0:]] == ["Sub", "Base", "Sub"]


def test_rows_reach_disk_without_close():
    import time

    path = os.path.join(os.environ["MRV2_TRACE_DIR"], f"mrv2_trace_{os.getpid()}.jsonl")
    mrv2_trace.tracer().write(ev="probe_flush")
    time.sleep(2.5)
    last = [json.loads(line) for line in open(path)][-1]
    assert last["ev"] == "probe_flush"
