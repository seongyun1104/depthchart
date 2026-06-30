# lmcache

LMCache wiring and hit-rate-controlled workload generation.

```
configs/cpu_only.yaml   HBM + in-process CPU (32 GB)        — Phase 1 main
configs/cpu_disk.yaml   HBM + CPU (32 GB) + local NVMe (200 GB)  — Falsifier #5
configs/cpu_redis.yaml  HBM + CPU (16 GB) + local Redis     — Phase 2 / parity
workload_gen.py         HitRateController — target-hit-rate prompt generator
```

LMCache is engaged via env var `LMCACHE_CONFIG_FILE` (SGLang has no
`--lmcache-config` flag). The runner exports it before launching SGLang.

## Backend choice trade-off

| backend         | hit latency       | sweep-cell cache persistence | TTFT noise              |
|-----------------|-------------------|------------------------------|-------------------------|
| in-process CPU  | ~10-100 μs        | no (cold on restart)         | none (cleanest signal)  |
| Local NVMe      | ~ms (disk I/O)    | yes                          | adds retrieval latency  |
| Local Redis     | ~1 ms (UDS/TCP)   | yes                          | adds retrieval latency  |

P1 main hypothesis test wants the cleanest TTFT signal → `cpu_only` is the
default. Falsifier #5 ("retrieval latency erases prefill-skip gain") needs
a slower tier to surface — that's what `cpu_disk` or `cpu_redis` are for.

Remote / multi-node Redis, Mooncake Store, S3, Aerospike, NIXL stay out
of P1 — they belong to the (excluded) multi-instance track.

## Hit rate control

Hit rate = fraction of prompts whose shared prefix has already been served
once in this run. `HitRateController` keeps a prefix bank and decides per
prompt whether to reuse (warm) or mint a new prefix (cold) to track the
target rate. Synthetic; ShareGPT replay is a later option.
