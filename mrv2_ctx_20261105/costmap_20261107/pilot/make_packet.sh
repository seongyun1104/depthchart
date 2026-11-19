#!/usr/bin/env bash
# Builds costmap_packet.tar from committed files only (box layout: /root/smoke, /root/costmap).
#   bash costmap_20261107/pilot/make_packet.sh      (from mrv2_ctx_20261105/)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain --untracked-files=no smoke costmap_20261107)" ] \
  || { echo "uncommitted changes under smoke/ or costmap_20261107/: commit first" >&2; exit 2; }
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/costmap/pilot" "$STAGE/costmap/stage1"
git ls-files smoke | grep -Ev '^smoke/run_2026' | while read -r f; do mkdir -p "$STAGE/$(dirname "$f")"; cp -p "$f" "$STAGE/$f"; done
for f in analyze_costmap.py costmap_cmd.py run_costmap.py replan.py pilot/run_pilot_box.sh stage1/run_stage1_box.sh stage1/zgate.py; do
  cp -p "costmap_20261107/$f" "$STAGE/costmap/$f"
done
( cd "$STAGE" && find smoke costmap -type f | LC_ALL=C sort | xargs shasum -a 256 > "$STAGE.files" )
mv "$STAGE.files" "$STAGE/costmap/pilot/PACKET_FILES.sha256"
shasum -a 256 "$STAGE/costmap/pilot/PACKET_FILES.sha256" | cut -d' ' -f1 > "$STAGE/costmap/pilot/PACKET_SHA256"
git rev-parse HEAD > "$STAGE/costmap/pilot/SOURCE_COMMIT"
tar -C "$STAGE" -cf costmap_20261107/pilot/costmap_packet.tar smoke costmap
echo "packet: costmap_20261107/pilot/costmap_packet.tar"
echo "files: $(wc -l < "$STAGE/costmap/pilot/PACKET_FILES.sha256")  manifest sha256: $(cat "$STAGE/costmap/pilot/PACKET_SHA256")  source commit: $(cat "$STAGE/costmap/pilot/SOURCE_COMMIT")"
shasum -a 256 costmap_20261107/pilot/costmap_packet.tar
