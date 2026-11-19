# Stage 1 (S1-odd) r2 — result (2026-11-11)

Registration: tag `costmap-stage1-r2-prereg-20261111` (`444ce5c`), packet manifest `dd8f8c69…3d4a`, engine tree `eeb37ef757`
(vLLM `bd09764c45`). Box: Vast instance 60184816 (m52521, H100 NVL 95,830 MiB, driver 595.91, $3.401/h), created ≈08:11Z,
running 08:18:58Z, ALL_DONE 09:46:07Z, collected 09:51:55Z, destroyed and confirmed absent 09:52:09Z (0 instances). Credit
10.1586 → 4.8393 (−$5.32; settlement may lag). Evidence `evidence/costmap_out/` (381 files, `EVIDENCE_SHA256SUMS`).

## Run
- Z gate (`zgate/zgate.json`): pass — startup 215 s, /health 200, no Traceback/TypeError, captures target 11 / drafter_prefill
  11, request 16 tokens, success +1, draft tokens +0, 16 scheduler rows all K=0, GPU 0 → 0 MiB, 1,160 default cache files
  unchanged, gate dir removed.
- 10/10 launches complete in the registered order (K3,K1,K5,OFF,Z,Z,OFF,K5,K1,K3); every early and launch gate passed,
  valid share 1.0; in-launch drift −0.023…+0.025. K3_r0 started cold (215 s), as registered.
- Report: box `costmap_report.json` == local re-run of `analyze_costmap.py` (identical JSON). All 10 shapes in the compared
  set for all three metrics; `not_compared` = the warm-up and two sentinel repeats only (by design).

## compare_stage1 verdicts (counts over the 10 shapes; pair = later / earlier config)
| pair | step period | delivered ms/token (primary) | server ms/token (secondary) |
|---|---|---|---|
| Z / OFF | 10 exceeds | 10 exceeds | 10 exceeds |
| K1 / OFF, K3 / OFF, K5 / OFF | 10 exceeds each | 10 exceeds each | 10 exceeds each |
| K1 / Z | 4 exceeds · 5 inconclusive · 1 within | 10 exceeds | 10 exceeds |
| K3 / Z, K5 / Z, K3 / K1, K5 / K1 | 10 exceeds each | 10 exceeds each | 10 exceeds each |
| K5 / K3 | 10 exceeds | 8 exceeds · 2 inconclusive (b16·1k, b32·2k) | 9 exceeds · 1 within |

Step period, as % over OFF (mean of the two launches, exp(log ratio) − 1): Z +3.4…+6.7 % at every shape; K1 +3.5…+6.3 %
except b16 (+16.3 % at 1k, +15.3 % at 4k); K3 +11.6…+42.8 %; K5 +19.2…+70.2 %, growing with BS and with context. K1 vs Z: step period
indistinguishable from launch spread at 6 of 10 shapes; the 4 that exceed are b1·4k, b1·16k (+0.5…+0.7 %) and b16·1k,
b16·4k (+10.4…+10.6 %). All ranges recomputed from `analysis/pairs.csv` by script.

## What this does and does not say
- Answers the registered question: per shape, which K-to-K step-cost differences exceed launch-to-launch spread. Almost
  all do; the exception is K1 vs Z (6/10 not), and the delivered-token K5 vs K3 at b16·1k and b32·2k (inconclusive).
- The Z configuration as a whole (drafter loaded, dynamic schedule K=0) is 3.4–6.7 % slower per step than OFF at every
  shape (attempt 2 had no Z). Not decomposed: memory split from loading the drafter, execution path, or per-step work
  that remains at K=0 — this is not "the cost of loading alone".
- `exceeds_aa` is the registered decision rule applied to two launches, not a significance test or a general
  reproducibility guarantee; `within_aa` / `inconclusive` mean the two launches could not separate the pair (K1 vs Z at
  6 of 10 shapes), not that there is no difference.
- Delivered / server ms per token depend on acceptance, which is 0.65–1.0 here because prompts are random tokens with
  ignore_eos and 640 output tokens; they are not a speedup estimate for real traffic.
- b16 K1 step jump: a reproduced outlier (both launches agree; target FULL/r16 and drafter FULL graphs as in the other
  configs). That lowers the chance of one-off noise but does not rule out a systematic measurement or condition
  difference; cause not established.
- Not claimed: speedup, optimal K, any policy ranking; K2/K4, BS 64 not run. Attempt 2 and pilots (other engine trees)
  are not merged or compared here.

Artifacts: `analysis/export_csv.py` → `cells.csv`, `pairs.csv`, `launches.csv` (dashboard sources);
`costmap_report_local.json`; public cost map: `public_20261119/COST_MAP.md`.
