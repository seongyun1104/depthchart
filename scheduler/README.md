# scheduler

SGLang server lifecycle for sweep cells, plus analysis notes on the SGLang
scheduler's behavior under LMCache + spec.

```
sglang_runner.py       SGLangServer — one server process per (model, spec, lmcache)
notes/                 source-reading notes (TBD)
patches/               candidate scheduler patches for OSS PRs (TBD)
```

## SGLang scheduler entry

`python/sglang/srt/managers/scheduler.py` — request batching, prefix-cache
matching, schedule policy. RadixAttention + longest-prefix-match already
chooses cache-friendly batch composition; the open question is whether the
cache-hit decision should also feed into the speculative-on/off policy.

This dir holds the reading notes and any patch experiments we may upstream
once the hypothesis sweep produces a clear signal.

## Falsifier #1 cross-check

Falsifier #1 (prefill slack ≠ decode slack unless phases mix) hinges on
SGLang's chunked-prefill behavior. Verify by reading `scheduler.py` +
prefill chunking logic before claiming the hypothesis applies to
PD-disaggregated configurations (it does not — by design we stay
non-disaggregated).
