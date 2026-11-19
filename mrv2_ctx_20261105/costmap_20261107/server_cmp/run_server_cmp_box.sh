#!/usr/bin/env bash
# Box side of the server comparison (kernel_probe_20261117/server_cmp/PREREG_SERVER_66852_DRAFT_20261119.md): r2
# provision and preflight unchanged, then the built _C_stable_libtorch swapped in and checked (hash on disk, hash
# loaded, one op), then one warm launch and six K1 launches 0 P P 0 0 P on the stage-1 plan. Every exit leaves a
# marker in /root/costmap_out: ALL_DONE or STOPPED (runner), or STOPPED with BOX_STEP_FAILED / SO_* from this script.
#   DEADLINE_EPOCH=<epoch> bash /root/costmap/server_cmp/run_server_cmp_box.sh
set -euo pipefail
: "${DEADLINE_EPOCH:?}"
OUT=/root/costmap_out
SC=/root/costmap/server_cmp
SO=/workspace/vllm/vllm/_C_stable_libtorch.abi3.so
mkdir -p "$OUT"
step=start
trap 'rc=$?; [ -e "$OUT/ALL_DONE" ] || [ -e "$OUT/STOPPED" ] || printf "BOX_STEP_FAILED %s rc=%s\n" "$step" "$rc" > "$OUT/STOPPED"' EXIT
log() { echo "$* $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"; }
stop() { printf '%s\n' "$1" > "$OUT/STOPPED"; log "$1"; exit 3; }
echo "PACKET_SHA256 $(cat /root/costmap/pilot/PACKET_SHA256)" | tee "$OUT/box.log"
CAND_SHA=$(cat "$SC/CAND_SO_SHA256")
step=packet_so
[ "$(sha256sum "$SC/cand_C_stable_libtorch.abi3.so" | cut -c1-64)" = "$CAND_SHA" ] || stop "SO_PACKET_MISMATCH"
step=provision; log PROVISION_START
PATCH=/root/smoke/box/bs_ctx_bd09764c45.patch TRANSFORMERS_PIN=5.22.1 bash /root/smoke/box/provision_mrv2.sh >> "$OUT/box.log" 2>&1
step=preflight; log PREFLIGHT_START
SMOKE=/root/smoke bash /root/smoke/box/preflight_mrv2.sh --download >> "$OUT/box.log" 2>&1
step=so_swap; log SO_SWAP_START
WHEEL_SHA=$(sha256sum "$SO" | cut -c1-64)
[ "$WHEEL_SHA" != "$CAND_SHA" ] || stop "SO_WHEEL_EQUALS_CAND"
cp "$SO" /root/wheel_C_stable_libtorch.abi3.so
cp "$SC/cand_C_stable_libtorch.abi3.so" "$SO"
[ "$(sha256sum "$SO" | cut -c1-64)" = "$CAND_SHA" ] || stop "SO_SWAP_MISMATCH"
printf 'wheel %s\ncand %s\n' "$WHEEL_SHA" "$CAND_SHA" > "$OUT/so_sha256.txt"
step=so_load
/venv/main/bin/python "$SC/so_load_check.py" "$CAND_SHA" > "$OUT/so_load.json" 2>> "$OUT/box.log" || stop "SO_LOAD_FAILED"
log SO_LOAD_PASS
CONFIGS=K1,K1,K1,K1,K1,K1
ARMS=0,P,P,0,0,P
step=compare; log "COMPARE_START configs=$CONFIGS arms=$ARMS"
set +e
PATH=/venv/main/bin:$PATH CONFIGS="$CONFIGS" ARMS="$ARMS" WARM_ARM=0 ARM_SO_SHA256="$CAND_SHA" PLAN=stage1 PLAN_SEED=20261109 \
  LAUNCH_EST_S=900 DEADLINE_EPOCH="$DEADLINE_EPOCH" MODEL_DIR=/root/models/target DRAFT_DIR=/root/models/draft \
  SMOKE=/root/smoke COSTMAP=/root/costmap OUT="$OUT" /venv/main/bin/python /root/costmap/run_costmap.py >> "$OUT/box.log" 2>&1
rc=$?
set -e
log "COMPARE_END rc=$rc"
step=report
# analyze_costmap.report keys launches by (config, rep) and would merge the two arms; per-launch reports are in each dir
(cd /root/costmap && /venv/main/bin/python compare_arms.py "$OUT" > "$OUT/compare_arms.json" 2>> "$OUT/box.log") || true
exit "$rc"
