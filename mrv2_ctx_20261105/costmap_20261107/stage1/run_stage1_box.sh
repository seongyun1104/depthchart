#!/usr/bin/env bash
# Box side of stage 1 (STAGE1_ODD_PREREG.md; Z gate per STAGE1_R2_PREREG_DRAFT_20261111.md §4): OFF, Z, K1, K3, K5, each twice (A order, then reversed). Every exit leaves a marker in /root/costmap_out:
# ALL_DONE or STOPPED (runner), or STOPPED with BOX_STEP_FAILED from this script.
#   DEADLINE_EPOCH=<smoke_deadline_epoch.txt> bash /root/costmap/stage1/run_stage1_box.sh
set -euo pipefail
: "${DEADLINE_EPOCH:?}"
OUT=/root/costmap_out
mkdir -p "$OUT"
step=start
trap 'rc=$?; [ -e "$OUT/ALL_DONE" ] || [ -e "$OUT/STOPPED" ] || printf "BOX_STEP_FAILED %s rc=%s\n" "$step" "$rc" > "$OUT/STOPPED"' EXIT
echo "PACKET_SHA256 $(cat /root/costmap/pilot/PACKET_SHA256)" | tee "$OUT/box.log"
step=provision; echo "PROVISION_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
PATCH=/root/smoke/box/bs_ctx_bd09764c45.patch TRANSFORMERS_PIN=5.22.1 bash /root/smoke/box/provision_mrv2.sh >> "$OUT/box.log" 2>&1
step=preflight; echo "PREFLIGHT_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
SMOKE=/root/smoke bash /root/smoke/box/preflight_mrv2.sh --download >> "$OUT/box.log" 2>&1
step=zgate; echo "ZGATE_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
set +e
DEADLINE_EPOCH="$DEADLINE_EPOCH" MODEL_DIR=/root/models/target DRAFT_DIR=/root/models/draft SMOKE=/root/smoke COSTMAP=/root/costmap \
  OUT="$OUT/zgate" ZGATE_DIR=/root/zgate PATH=/venv/main/bin:$PATH /venv/main/bin/python /root/costmap/stage1/zgate.py >> "$OUT/box.log" 2>&1
rc=$?
set -e
if [ "$rc" != 0 ]; then
  reason=$(/venv/main/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason") or "no reason")' "$OUT/zgate/zgate.json" 2>/dev/null || echo "no zgate.json")
  printf 'ZGATE rc=%s %s\n' "$rc" "$reason" > "$OUT/STOPPED"
  echo "ZGATE_FAILED rc=$rc $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
  exit 3
fi
echo "ZGATE_PASS $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
step=stage1; echo "STAGE1_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
set +e
PATH=/venv/main/bin:$PATH CONFIGS=K3,K1,K5,OFF,Z,Z,OFF,K5,K1,K3 PLAN=stage1 PLAN_SEED=20261109 LAUNCH_EST_S=900 DEADLINE_EPOCH="$DEADLINE_EPOCH" \
  MODEL_DIR=/root/models/target DRAFT_DIR=/root/models/draft SMOKE=/root/smoke \
  COSTMAP=/root/costmap OUT="$OUT" /venv/main/bin/python /root/costmap/run_costmap.py >> "$OUT/box.log" 2>&1
rc=$?
set -e
echo "STAGE1_END rc=$rc $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
step=report
/venv/main/bin/python /root/costmap/analyze_costmap.py "$OUT" > "$OUT/costmap_report.json" 2>> "$OUT/box.log" || true
exit "$rc"
