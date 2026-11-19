#!/usr/bin/env bash
# MRv2 smoke rental (SMOKE_PACKET.md): check the offer, create ONE instance,
# verify, arm the laptop watchdog. Run only on the user's explicit go for one
# offer and ceiling; files go to a fresh run directory:
#   RUN_DIR=<new dir> bash create_and_arm_smoke.sh <OFFER_ID>   (CEILING=<usd> optional)
# Creates nothing unless the offer is listed now, not deverified (unverified only
# with ALLOW_UNVERIFIED=1), driver >= 580, and the expected time on a fast host
# x 1.2 fits the deadline left after reserving the transfer cost (55 GB x $/GB).
# Writes smoke_deadline_epoch.txt = deadline - 300 s, run_smoke.sh's DEADLINE_EPOCH.
# Copied from e1_run_20261029/create_and_arm.sh; changed: CEILING defaults to the
# live credit less 0.30, FAST_MIN for the trimmed smoke, run directory and watchdog by
# absolute path.
# No set -x: the create response carries an API key.
set -euo pipefail
WATCHDOG=/Users/kimseongyun/PycharmProjects/CodePython/depthchart/restart_run_20261027/watchdog.sh
[ -x "$WATCHDOG" ] || [ -f "$WATCHDOG" ] || { echo "watchdog script missing: $WATCHDOG" >&2; exit 2; }
: "${RUN_DIR:?set RUN_DIR}"
# No user ceiling (2026-11-05): the bound is the account credit read now, less a
# margin so the balance does not go negative.
if [ -z "${CEILING:-}" ]; then
  CEILING=$(vastai show user --raw | python3 -c 'import json,sys;u=json.load(sys.stdin);print(round(float(u["credit"])+float(u.get("balance") or 0)-0.30,4))')
  echo "CEILING from live credit: $CEILING"
fi
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"

OFFER=${1:?usage: create_and_arm.sh <OFFER_ID>}
ALLOW_UNVERIFIED=${ALLOW_UNVERIFIED:-0}
GB_DOWN=55                  # models 37.0 GB + wheel 0.3 GB + pip + image, upper estimate
FAST_MIN=${FAST_MIN:-50}    # SMOKE_PACKET.md: provision 15 + L1 8 + L2 13 + L4 9 + collect 5
RESERVE=0.40                # estimated margin for unexplained charges, not a guarantee
R_CONS_FACTOR=1.05          # watchdog uses offer dph x 1.05 until the billed rate is read
[ ! -e instance_id.txt ] || { echo "instance_id.txt exists: a create already ran here" >&2; exit 2; }

# 1. offer listed now and meets the criteria (read only)
# The listing varies between queries, so both query forms are merged.
OFFERS=$( { vastai search offers 'gpu_name=H100_NVL num_gpus=1' --raw
            vastai search offers 'gpu_name=H100_NVL num_gpus=1 verified=any' --raw; } \
  | python3 -c '
import json, sys
dec, txt, out, i = json.JSONDecoder(), sys.stdin.read(), [], 0
while i < len(txt):
    if txt[i].isspace():
        i += 1; continue
    obj, i = dec.raw_decode(txt, i); out += obj
print(json.dumps(out))')
read -r DPH DOWN < <(printf '%s' "$OFFERS" | python3 -c '
import json, sys
offer, allow_unv = int(sys.argv[1]), sys.argv[2] == "1"
cap, reserve, factor, gb, fast = (float(x) for x in sys.argv[3:8])
o = [x for x in json.load(sys.stdin) if x.get("id") == offer]
if not o:
    sys.exit("offer not listed")
o = o[0]
v, why = o.get("verification"), []
if v == "deverified" or (v != "verified" and not allow_unv):
    why.append("verification %s" % v)
if "NVL" not in (o.get("gpu_name") or ""):
    why.append("gpu %s" % o.get("gpu_name"))
if int(str(o.get("driver_version") or "0").split(".")[0]) < 580:
    why.append("driver %s" % o.get("driver_version"))
r, down = o["dph_total"], o.get("inet_down_cost") or 0
rc = r * factor
avail = (cap - reserve - down * gb - 5 / 60 * rc) / rc * 60
if avail < fast * 1.2:
    why.append("deadline %.1f min < %.1f min needed (fast host x 1.2)" % (avail, fast * 1.2))
if why:
    sys.exit("; ".join(why))
print(r, down)' "$OFFER" "$ALLOW_UNVERIFIED" "$CEILING" "$RESERVE" "$R_CONS_FACTOR" "$GB_DOWN" "$FAST_MIN") \
  || { echo "offer $OFFER rejected; nothing created" >&2; exit 3; }
[ -n "${DPH:-}" ] || { echo "offer $OFFER rejected; nothing created" >&2; exit 3; }
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  python3 -c "
r=$DPH*$R_CONS_FACTOR; d=$DOWN*$GB_DOWN
print('offer $OFFER passes: dph=$DPH transfer=%.2f usd minutes_until_deadline=%.1f need=%.1f' % (d, ($CEILING-$RESERVE-d-5/60*r)/r*60, $FAST_MIN*1.2))"
  exit 0
fi

# 2. create; conservative start = before the command
START=$(date +%s)
echo "$START" > command_start_epoch.txt
set +e
RAW=$(vastai create instance "$OFFER" --image vastai/base-image:cuda-12.8.1-auto \
      --disk 100 --ssh --direct --raw 2>/dev/null)
RC=$?
set -e
# Only allowlisted fields leave this variable; the raw response is never written.
ID=$(printf '%s' "$RAW" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
except Exception:
    print(""); sys.exit(0)
ok = d.get("success") is True and isinstance(d.get("new_contract"), int)
print(d["new_contract"] if ok else "")')
unset RAW
if [ "$RC" -ne 0 ] || [ -z "$ID" ]; then
  echo "create failed or response not understood (rc=$RC). Check for a stray instance NOW:" >&2
  echo "  vastai show instances-v1 --raw | python3 -c \"import json,sys;d=json.load(sys.stdin);print(d.get('total_instances'),[i['id'] for i in d.get('instances',[])])\"" >&2
  exit 4
fi
echo "$ID" > instance_id.txt
printf '{"success": true, "new_contract": %s, "offer": %s, "offer_dph_precheck": %s, "inet_down_cost": %s, "command_start_epoch": %s}\n' \
  "$ID" "$OFFER" "$DPH" "$DOWN" "$START" > create_allowlisted.json

destroy_now() {   # $1 = reason; destroys and re-queries until the id is gone
  echo "DESTROYING $ID: $1" >&2
  for i in 1 2 3 4 5 6 7 8 9 10; do
    yes | vastai destroy instance "$ID" > /dev/null 2>&1 || true
    sleep 10
    vastai show instances-v1 --raw | python3 -c '
import json, sys
sys.exit(0 if int(sys.argv[1]) not in [i["id"] for i in json.load(sys.stdin).get("instances", [])] else 1)' "$ID" \
      && { echo "destroyed $ID (confirmed absent)" >&2; return 0; }
  done
  echo "DESTROY NOT CONFIRMED for $ID - check now: vastai show instances-v1 --raw" >&2
  return 1
}

# 3. watchdog first, from the conservative rate (earlier deadline than the billed rate gives)
R=$(python3 -c "print(round($DPH * $R_CONS_FACTOR, 4))")
D=$(python3 -c "print(int($START + ($CEILING - $RESERVE - $DOWN*$GB_DOWN - 5/60*$R) / $R * 3600))")
W=$((D + 240))
nohup caffeinate -i bash "$WATCHDOG" "$ID" "$W" "$PWD/watchdog.log" \
  > /dev/null 2>&1 < /dev/null & disown
sleep 2
if ! grep -q armed watchdog.log 2>/dev/null; then destroy_now "watchdog not armed" || true; exit 5; fi
printf 'conservative_rate=%s deadline_epoch=%s watchdog_epoch=%s\n' "$R" "$D" "$W" > deadline_conservative.txt
echo $((D - 300)) > smoke_deadline_epoch.txt

# 4. billed rate and transfer price of the instance itself: the deadline above holds
# only if neither exceeds what it assumed; otherwise, or if unreadable, destroy.
CHECK=""
for i in 1 2 3 4 5 6; do
  CHECK=$(vastai show instances-v1 --raw | python3 -c '
import json, sys
iid, r, down = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for i in json.load(sys.stdin).get("instances", []):
    if i.get("id") == iid and i.get("dph_total"):
        d = i.get("inet_down_cost") or 0
        print("ok" if i["dph_total"] <= r and d <= down else "over billed=%s down=%s" % (i["dph_total"], d))' \
    "$ID" "$R" "$DOWN") || true
  [ -n "$CHECK" ] && break
  sleep 10
done
if [ "$CHECK" != ok ]; then
  # the watchdog stays armed unless the destroy is confirmed
  if destroy_now "billed rate check: ${CHECK:-unreadable}"; then pkill -f "watchdog.sh $ID" || true; fi
  exit 6
fi

# 5. report (allowlisted fields only)
vastai show instances-v1 --raw | python3 -c '
import json, sys
for i in json.load(sys.stdin).get("instances", []):
    print({k: i.get(k) for k in ("id", "actual_status", "dph_total")})'
echo "instance $ID created; watchdog armed for $(date -u -r "$W" +%FT%TZ) (deadline $(date -u -r "$D" +%FT%TZ), rate assumed $R)"
echo "manual stop: yes | vastai destroy instance $ID && vastai show instances-v1 --raw && pkill -f \"watchdog.sh $ID\""
