#!/usr/bin/env bash
# Smoke provisioning: vLLM main 7ecc83a987 + the local BS x ctx patch (carried as a
# file, never pushed), precompiled binaries from the base commit's wheel.
# Derived from e1_prep/box/provision_b1.sh. Markers only after checked commands.
#
#   PATCH=/root/smoke/box/bs_ctx_bd09764c45.patch bash provision_mrv2.sh
# Local dry run (no install, no wheel download; fetch, apply and tree checks are real):
#   DRY_RUN=1 WS=<dir> OUT=<dir> PATCH=<file> bash provision_mrv2.sh
set -euo pipefail
PY=${PY:-/venv/main/bin/python}
PIP=${PIP:-/venv/main/bin/pip}
WS=${WS:-/workspace}
OUT=${OUT:-/root/preflight}
WHEEL_DIR=${WHEEL_DIR:-/root/wheels}
DRY_RUN=${DRY_RUN:-0}
: "${PATCH:?}"

BASE=7ecc83a9877552a5f4be31c133ab8502dd40f50e
BASE_TREE=d11bc4e60d5cef90d5b2385ff6b8a985269ff2e7
PATCHED_TREE=eeb37ef75784c994c61e4d3d74b183cd0c395d72
PATCH_SHA256=85f25f7e242226fe71e47476df3e1312d28819d01cc569497d82854c366278d7
WHEEL_NAME=vllm-0.33.0rc1.dev784+g7ecc83a98-cp38-abi3-manylinux_2_28_x86_64.whl
WHEEL_URL="https://wheels.vllm.ai/$BASE/vllm-0.33.0rc1.dev784%2Bg7ecc83a98-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_BYTES=317037303
TRANSFORMERS_PIN=${TRANSFORMERS_PIN:?set to the version the config contract passed with}

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
[ "$(sha "$PATCH")" = "$PATCH_SHA256" ] || { echo "PATCH_SHA_MISMATCH"; exit 1; }

mkdir -p "$WS" "$OUT"
t0=$(date +%s)
cd "$WS"
rm -rf vllm && mkdir vllm && cd vllm
git init -q .
git remote add origin https://github.com/vllm-project/vllm
git fetch -q --depth 1 origin "$BASE"
git checkout -q --detach FETCH_HEAD
[ "$(git rev-parse HEAD)" = "$BASE" ] || { echo "BASE_MISMATCH"; exit 1; }
[ "$(git rev-parse 'HEAD^{tree}')" = "$BASE_TREE" ] || { echo "BASE_TREE_MISMATCH"; exit 1; }
git apply --index "$PATCH"
[ "$(git write-tree)" = "$PATCHED_TREE" ] || { echo "PATCHED_TREE_MISMATCH"; exit 1; }
git diff --cached --name-only > "$OUT/patch_files.txt"
if grep -qvE '\.py$' "$OUT/patch_files.txt"; then
  echo "PATCH_TOUCHES_NON_PYTHON:"; cat "$OUT/patch_files.txt"; exit 1
fi
echo "base=$BASE patched_tree=$PATCHED_TREE files=$(wc -l < "$OUT/patch_files.txt")"
t1=$(date +%s); echo "fetch_seconds=$((t1-t0))"

if [ "$DRY_RUN" = 1 ]; then
  code=$(curl -sI -o /dev/null -w '%{http_code}' "$WHEEL_URL")
  len=$(curl -sI "$WHEEL_URL" | tr -d '\r' | awk 'tolower($1)=="content-length:" {print $2}')
  [ "$code" = 200 ] && [ "$len" = "$WHEEL_BYTES" ] || { echo "WHEEL_HEAD_MISMATCH code=$code len=$len"; exit 1; }
  echo "dry_run: wheel listed with $len bytes; install skipped"
  echo "PROVISION_DRY_RUN_DONE"
  exit 0
fi

mkdir -p "$WHEEL_DIR"
curl -sS -L --fail --retry 3 -o "$WHEEL_DIR/$WHEEL_NAME" "$WHEEL_URL"
[ "$(stat -c %s "$WHEEL_DIR/$WHEEL_NAME")" = "$WHEEL_BYTES" ] || { echo "WHEEL_SIZE_MISMATCH"; exit 1; }
sha256sum "$WHEEL_DIR/$WHEEL_NAME" | tee "$OUT/wheel_sha256.txt"
t2=$(date +%s); echo "wheel_seconds=$((t2-t1))"

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_DIR/$WHEEL_NAME"
"$PIP" install -v -e . > /root/install_vllm.log 2>&1
grep -q "Using user-specified precompiled wheel location: $WHEEL_DIR/$WHEEL_NAME" /root/install_vllm.log \
  || { echo "WHEEL_LOCATION_NOT_USED"; exit 1; }
t3=$(date +%s); echo "install_seconds=$((t3-t2))"

"$PIP" install -q "transformers==$TRANSFORMERS_PIN"
INSTALLED=$("$PY" -c 'import transformers;print(transformers.__version__)')
[ "$INSTALLED" = "$TRANSFORMERS_PIN" ] || { echo "TRANSFORMERS_PIN_FAILED got=$INSTALLED"; exit 1; }
"$PIP" check > "$OUT/pip_check.txt" 2>&1 || { echo "PIP_CHECK_FAILED"; cat "$OUT/pip_check.txt"; exit 1; }
echo "transformers=$INSTALLED pip_check=ok"

"$PY" "$(dirname "$0")/wheel_provenance.py" "$WHEEL_DIR/$WHEEL_NAME" "$WS/vllm" > "$OUT/wheel_provenance.json"
"$PIP" freeze | grep -iE '^(torch|torchvision|vllm|transformers|flashinfer)' | tee "$OUT/pins.txt"
echo "PROVISION_DONE"
