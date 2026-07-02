# lmcache

LMCache backend configs + hit-rate-controlled workload generation for the
vLLM `LMCacheConnectorV1` path.

## Two backend cases

The lab pins the backend to the shape of KV cache sharing that matters:

| Case | File | When it applies |
|---|---|---|
| **CPU DRAM local tier** (single instance) | `configs/cpu_only.yaml` | In-process CPU offload; fastest sharing when one server owns the entire prefix cache and has DRAM headroom. **P1 main.** |
| **Cross-instance shared tier** (Redis) | `configs/cpu_redis.yaml` | Multiple vLLM instances must share the same KV cache; Redis is the correct remote backend for KV cache sharing. Phase 2 / parity. |
| Local NVMe (falsifier only) | `configs/cpu_disk.yaml` | Falsifier #5 — measure whether retrieval latency erases prefill-skip gain. |

## Schema (LMCache `v1/config.py`)

vLLM engages LMCache via `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
and reads its own YAML from `LMCACHE_CONFIG_FILE` (env var exported by
`scheduler/vllm_runner.py`). Fields used here match
`_CONFIG_DEFINITIONS` in `lmcache/v1/config.py`:

- `chunk_size: int`
- `local_cpu: bool`, `max_local_cpu_size: float` (GB)
- `local_disk: Optional[str]` (e.g. `"file:///tmp/lmcache_storage"`),
  `max_local_disk_size: float` (GB)
- `remote_url: Optional[str]` (`"redis://host:port"`)
- `remote_serde: Optional[str]`
- `save_decode_cache: bool`, `enable_blending: bool`

## Backend trade-off

| backend         | hit latency       | cross-instance sharing | TTFT noise             |
|-----------------|-------------------|------------------------|------------------------|
| in-process CPU  | ~10-100 μs        | no                     | none (cleanest signal) |
| Local NVMe      | ~ms (disk I/O)    | no                     | adds retrieval latency |
| Local Redis     | ~1 ms (UDS/TCP)   | yes                    | adds retrieval latency |

P1's main hypothesis test wants the cleanest TTFT signal → `cpu_only` is
the default. Multi-instance sharing tests use `cpu_redis`.

Remote / multi-node Redis, Mooncake Store, S3, Aerospike, NIXL stay out
of P1.

## Hit rate control

`HitRateController.next()` produces prompts whose shared prefix is either
reused (warm) or freshly minted (cold) so the running warm/total ratio
tracks `target_hit_rate`. Synthetic; ShareGPT replay is a later option.
