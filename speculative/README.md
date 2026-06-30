# speculative

Speculative method adapter and acceptance-rate hooks.

```
adapter.py       SpecConfig + SGLang flag translation; for_exaone_45(k) helper
acceptance.py    AcceptanceTrace — per-step drafted/accepted counters
```

## EXAONE 4.5 spec route

EXAONE 4.5 ships a native MTP head (model card: "64 Main layers + 1 MTP
layer"). Three SGLang paths are valid and we sweep over them:

- `--speculative-algorithm MTP` — direct route to the native MTP head
  (our default for EXAONE 4.5)
- `--speculative-algorithm NEXTN` — SGLang's original "single MTP module
  as draft" path; behaves equivalently for one MTP layer
- `--speculative-algorithm EAGLE` — what the official EXAONE model card
  prescribes for SGLang. Kept as fallback in case the MTP path is still
  hard-coded for DeepSeek in our SGLang version

No separate draft checkpoint is shipped — the spec head is part of the
target. `draft_model=None`. FastMTP / DFlash stay parked until the MTP
baseline is clean.

`num_steps` = number of draft tokens proposed per verification step.
`eagle_topk` = beam width per draft step. `num_draft_tokens` = max draft
tokens per spec attempt. (SGLang requires these three set together or all
left unset for auto-tune.)

## DFlash

DFlash lives in SGLang Spec V2 (PRs #22077, #23000). Plugging it in is a
later sweep, after the EAGLE3 baseline run is clean.
