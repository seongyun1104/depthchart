#!/usr/bin/env bash
# Rebuild the r2 GPU packet from this public branch and check it against the registered manifest
# (STAGE1_R2_PREREG.md §8: 24 files, manifest sha256 dd8f8c69…3d4a). Same layout and hashing as
# costmap_20261107/pilot/make_packet.sh at tag costmap-stage1-r2-prereg-20261111; that script needs the full lab
# history, this one only needs this branch. analyze_costmap.py and costmap_cmd.py come from repro_r2_tag/ (the
# registered versions); every other file is at its registered path.
#   bash public_20261119/build_r2_packet.sh [out.tar]      (from mrv2_ctx_20261105/)
set -euo pipefail
cd "$(dirname "$0")/.."
REGISTERED=dd8f8c6964a2cd866921100b3011b03e8a603d720bad54a5ca4c12b9242c3d4a
OUT_TAR=${1:-costmap_packet_r2.tar}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/costmap/pilot" "$STAGE/costmap/stage1"
find smoke -type f ! -path 'smoke/run_2026*' ! -name '*.pyc' ! -path '*/__pycache__/*' | while read -r f; do
  mkdir -p "$STAGE/$(dirname "$f")"; cp -p "$f" "$STAGE/$f"
done
for f in run_costmap.py replan.py pilot/run_pilot_box.sh stage1/run_stage1_box.sh stage1/zgate.py; do
  cp -p "costmap_20261107/$f" "$STAGE/costmap/$f"
done
cp -p public_20261119/repro_r2_tag/analyze_costmap.py public_20261119/repro_r2_tag/costmap_cmd.py "$STAGE/costmap/"
( cd "$STAGE" && find smoke costmap -type f | LC_ALL=C sort | xargs shasum -a 256 > "$STAGE.files" )
mv "$STAGE.files" "$STAGE/costmap/pilot/PACKET_FILES.sha256"
shasum -a 256 "$STAGE/costmap/pilot/PACKET_FILES.sha256" | cut -d' ' -f1 > "$STAGE/costmap/pilot/PACKET_SHA256"
tar -C "$STAGE" -cf "$OUT_TAR" smoke costmap
got=$(cat "$STAGE/costmap/pilot/PACKET_SHA256")
echo "files: $(wc -l < "$STAGE/costmap/pilot/PACKET_FILES.sha256")  manifest: $got"
[ "$got" = "$REGISTERED" ] && echo "MATCHES registered manifest" || { echo "DIFFERS from registered $REGISTERED" >&2; exit 1; }
