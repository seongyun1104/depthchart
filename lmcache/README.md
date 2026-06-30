# lmcache

LMCache wiring and hit-rate-controlled workload generation.

```
configs/default.yaml   LMCache backend config (CPU tier on, NVMe off by default)
workload_gen.py        HitRateController — synthetic prompts with target hit rate
```

## Hit rate control

Hit rate is the fraction of prompts whose shared prefix has already been
served once in this run. The controller maintains a prefix bank and decides
per prompt whether to reuse (warm) or mint a new prefix (cold) to track the
target rate.

This is synthetic. Realistic distributions come later via ShareGPT replay
(out of skeleton scope).

## Tier sweep (later)

```
configs/cpu_only.yaml      HBM + CPU
configs/cpu_nvme.yaml      HBM + CPU + NVMe (Falsifier #5 target)
```
