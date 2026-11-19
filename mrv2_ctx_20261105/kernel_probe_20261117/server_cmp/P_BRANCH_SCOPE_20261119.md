# Arm P covers only #66852's (16, 32] branch (checked 2026-11-19, before the registration tag)

Question (user): #66852's diff also changes M ≤ 8 and M ≤ 16 (N > 1280, K ≥ 1024). If arm P carried those branches,
b1 (M 2) and b8 (M 16) would not be controls and the K3 projection range would be wrong.

| source | what it shows |
|---|---|
| upstream diff (`gh pr diff 66852`, live 2026-11-19) | three new configs: `M8_N8192_K1024` (64×8×512), `M16_N8192_K1024` (64×16×512) inside `m <= 16` under `k >= 1024`; `M32_N8192_K1024` (64×32×512) inside `m <= 64` under `m <= 32 && k >= 1024` |
| local patch `gate2/sm90_fp8_candidates.patch` (sha256 `0bab9ba8…`, last changed `a9f035f`, before the build) | one hunk in the dispatch for #66852: inside `else if (m <= 64)`, `if (p66852 == 1 && m <= 32 && n > 1280 && k >= 1024)` → `sm90_fp8_pr66852_M32` (64×32×512). **No hunk in the `m <= 16` block**; its two upstream configs are not ported |
| built object (`hs_build/run2_l40_ok/evidence/logs/kernel_combos.txt`, from `scaled_mm_sm90_fp8.cu.o` of the .so `f2b60575…`) | 11 tile/cluster/schedule combos; `64x32x512` present, **no `64x8x512` or `64x16x512`** — the build cannot run upstream's M ≤ 16 configs |
| gate 2 kernel record (`gate2/run_g2/evidence/costmap_out/identity_{P,0}.jsonl`, gate_up 43008 and N 2048) | P differs from arm 0 only at M 17 and 32 (64×32×512 vs 64×64×256); at M 16, 33, 64, 65, 96, 128, 129 the kernel is the same. M 2 / 8 were not run in gate 2; for them the code and the object above are the evidence |

Consequence: in this build, P = #66852's (16, 32] branch alone. b1 (M 2) and b8 (M 16) stay controls; the K3
projection changes only b8 (M 32). The server comparison says nothing about #66852's M ≤ 16 branches; any
statement names "#66852's (16, 32] config", not "#66852".
