# K-rank flips across context in r2 — result: `none` (2026-11-19)

Rules `RULES.md` and script `k_rank.py` were committed before computing (`ee0c042`). Output: `RESULT.json`.

- **Pair flips:** none. No pair among Z, K1, K3 and K5 is resolved (`exceeds_aa`) in opposite directions at two
  contexts of the same batch.
- **Best option by delivered ms per token:** K5 at 8 of 10 shapes, resolved at each of them. At b16·1k and b32·2k,
  K3 and K5 are not separated (`inconclusive`, |log ratio| 0.011–0.013), and the two launches pick different ones.
  No batch has resolved best options that differ across contexts ⇒ **no best-K change**.

| batch | 1k | 4k | 16k / 2k |
|---|---|---|---|
| b1 | K5 (log K3/K5 0.33) | K5 (0.31) | K5 (0.28) |
| b8 | K5 (0.34) | K5 (0.24) | K5 (0.06) |
| b16 | K3 ≈ K5 (inconclusive) | K5 (0.08) | — |
| b32 | K5 (0.12) | K3 ≈ K5 at 2k (inconclusive) | |

Observed, not a candidate under the rules: at b8 the K5-over-K3 margin narrows with context (0.34 → 0.24 → 0.06). The
order does not reverse inside the grid.

Reading limits:
- K5 is the largest K run, so "best = K5" may sit at the edge of the grid.
- Acceptance here is 0.65–1.0 with random-token prompts, which favours large K. Real-traffic acceptance could move
  the order, and r2 cannot say which way.
- The outcome holds for this grid, model and prompts only. It is not evidence that a context axis cannot help
  elsewhere.

Next under the plan: with no candidate, no server re-measurement is proposed from r2.
