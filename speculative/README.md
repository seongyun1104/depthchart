# speculative

Speculative method adapter and acceptance-rate hooks.

```
adapter.py       SpecConfig + SGLang flag translation; for_exaone_45(k) helper
acceptance.py    AcceptanceTrace — per-step drafted/accepted counters
```

## EXAONE 4.5 spec route

EXAONE 4.5 MTP head ≡ EAGLE-style draft head (per `EAGLEWorkerV2` routing in
vLLM; same shape in SGLang). So the primary spec method for this lab is
`eagle3`. NEXTN is reserved for DeepSeek-V3-class targets.

Verify before first run:
1. SGLang accepts `--speculative-algorithm EAGLE3` for EXAONE 4.5 FP8
2. EAGLE3 draft head is shipped in the FP8 repo or needs separate fetch
3. `num-steps` semantics (does K=3 mean 3 drafted tokens per forward?)

## DFlash

DFlash lives in SGLang Spec V2 (PRs #22077, #23000). Plugging it in is a
later sweep, after the EAGLE3 baseline run is clean.
