# K-rank flips across context in r2 — rules fixed before computing (2026-11-19)

Question: at a fixed batch, when only the context changes, does the order of K options by **delivered ms per
committed token** reverse, and does it reverse in both launches? This is the first filter for the context axis
(#48944). It uses existing r2 data only (`costmap_20261107/run_20261111r2/analysis/{cells,pairs}.csv`), with no
new measurement.

Options ranked: Z (K = 0, drafter loaded), K1, K3, K5. OFF is reported but not ranked: it means no drafter, which
is a different deployment from the controller's choice set.
Contexts per batch: b1 and b8 at 1k / 4k / 16k, b16 at 1k / 4k, b32 at 1k / 2k.

## Pair flip
For options a, b and batch B:
- At context c, the order is **resolved** when `pairs.csv` (metric `delivered_ms_per_token`) gives `exceeds_aa` for
  the pair. This is the registered r2 rule: |log ratio| beyond the two-launch spread. The sign of `log_ratio` gives
  the direction.
- A **flip** is two contexts of the same batch where the pair is resolved in both and the directions are opposite.
- A flip is **repeated** when, in addition, each launch on its own (rep 0 vs rep 0, rep 1 vs rep 1, from
  `cells.csv`) shows the same opposite directions at the two contexts.

## Best option
For each (B, c), the best option is the one with the lowest mean delivered ms per token. It is **resolved** when
the best-vs-second pair is `exceeds_aa`. A **best-K change** is a batch where resolved best options differ across
contexts, and it is **repeated** when each launch on its own picks the same best options at those contexts.

## Outcome labels
- `candidate`: at least one repeated pair flip or repeated best-K change. Each one is listed with its batch,
  contexts and margins. A candidate is a reason to re-measure only that region on a server. It is not evidence
  that a context policy helps.
- `none`: no repeated flip or best-K change. The r2 grid then gives no context-axis candidate. That holds for this
  grid, model, prompts and acceptance (0.65–1.0 with random-token prompts) only.

## Not concluded either way
Real-traffic acceptance, contexts or batches outside the grid, or K2/K4 (not run). Two launches are the registered
rule, not a significance test.
