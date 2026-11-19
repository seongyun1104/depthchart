#!/usr/bin/env bash
# Box side of the K3 pilot (PILOT.md §3). Every exit leaves a marker in /root/costmap_out:
# ALL_DONE or STOPPED (runner), or STOPPED with BOX_STEP_FAILED from this script.
#   DEADLINE_EPOCH=<smoke_deadline_epoch.txt> bash /root/costmap/pilot/run_pilot_box.sh
set -euo pipefail
: "${DEADLINE_EPOCH:?}"
OUT=/root/costmap_out
mkdir -p "$OUT"
step=start
trap 'rc=$?; [ -e "$OUT/ALL_DONE" ] || [ -e "$OUT/STOPPED" ] || printf "BOX_STEP_FAILED %s rc=%s\n" "$step" "$rc" > "$OUT/STOPPED"' EXIT
echo "PACKET_SHA256 $(cat /root/costmap/pilot/PACKET_SHA256)" | tee "$OUT/box.log"
step=provision; echo "PROVISION_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
PATCH=/root/smoke/box/bs_ctx_8e5f0678a1.patch TRANSFORMERS_PIN=5.22.1 bash /root/smoke/box/provision_mrv2.sh >> "$OUT/box.log" 2>&1
step=preflight; echo "PREFLIGHT_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
SMOKE=/root/smoke bash /root/smoke/box/preflight_mrv2.sh --download >> "$OUT/box.log" 2>&1
step=pilot; echo "PILOT_START $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
set +e
PATH=/venv/main/bin:$PATH CONFIGS=K3 PLAN_SEED=20261107 DEADLINE_EPOCH="$DEADLINE_EPOCH" \
  MODEL_DIR=/root/models/target DRAFT_DIR=/root/models/draft SMOKE=/root/smoke \
  COSTMAP=/root/costmap OUT="$OUT" /venv/main/bin/python /root/costmap/run_costmap.py >> "$OUT/box.log" 2>&1
rc=$?
set -e
echo "PILOT_END rc=$rc $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"
step=report
/venv/main/bin/python /root/costmap/analyze_costmap.py "$OUT" > "$OUT/costmap_report.json" 2>> "$OUT/box.log" || true
exit "$rc"
