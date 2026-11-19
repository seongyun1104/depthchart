#!/usr/bin/env bash
# Smoke preflight on vLLM main 7ecc83a987 + the BS x ctx patch (MRv2). Derived from
# e1_prep/box/preflight_b1.sh; what changed: the engine pin (patched tree via
# git write-tree), the trace hook (smoke/trace), the harness checks. The config
# contract is the same: main also reads per-layer head sizes (gemma4.py:600,
# gemma4_mtp.py:269). Order is load-bearing: nothing large is downloaded
# until the runtime is proven to work, and the marker is written only after every
# required check has passed.
#
# Usage, on the instance (after provision_mrv2.sh):
#   bash preflight_mrv2.sh            # checks 1-4c only
#   bash preflight_mrv2.sh --download # checks 1-4c, then the model pull
#
# Credentials: the HF token is read from /root/.hf_token into a curl config
# file. No command that handles it runs under `set -x`.
set -euo pipefail

OUT=/root/preflight
LOG=$OUT/preflight.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

export TARGET_REV=1245be0199c1b792597ed7ea1ac5897269e66a89
export DRAFT_REV=96d4c8ca3cb38c107a8478587878124895d1e844
EXPECTED_HEAD=7ecc83a9877552a5f4be31c133ab8502dd40f50e
EXPECTED_TREE=eeb37ef75784c994c61e4d3d74b183cd0c395d72
PY=${PY:-/venv/main/bin/python}
PIP=${PIP:-/venv/main/bin/pip}
VLLM_DIR=/workspace/vllm
SMOKE=${SMOKE:-/root/smoke}

step() { printf '\n=== %s ===\n' "$1"; }
fail() { printf 'PREFLIGHT_FAIL: %s\n' "$1"; exit 1; }

step "1. shell discipline"
echo "set -euo pipefail is active; credential handling runs without -x"

step "2. versions"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not on PATH; this is not a usable GPU host"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/driver CUDA runtime: \1/p' | tee "$OUT/driver_cuda.txt"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRV_MAJOR=${DRV%%.*}
echo "driver major: $DRV_MAJOR"
"$PY" -V
"$PY" - <<'PYEOF' | tee "$OUT/versions.txt"
import json, torch
info = {
    "python": __import__("sys").version.split()[0],
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
}
# No torch API reports the installed driver's CUDA runtime: torch.cuda.driver_version
# and torch._C._cuda_getDriverVersion are both absent (checked on torch 2.14.0), and
# the hasattr guard that used to be here reported None every time, which is not a
# signal. The driver side comes from nvidia-smi above; torch reports its own build.
print(json.dumps(info, indent=2))
PYEOF
# Check the minimum driver a CUDA build requires. R580 is the documented
# minor-version compatibility floor for CUDA 13.x; a driver number and a CUDA
# version are not comparable quantities, so this is a floor lookup and nothing
# more. The floor is necessary and not sufficient, which is why steps 3 and 4
# run the runtime instead of trusting it.
TORCH_CUDA=$("$PY" -c 'import torch;print(torch.version.cuda or "none")')
echo "torch CUDA build: $TORCH_CUDA, host driver: $DRV"
case "$TORCH_CUDA" in
  13.*) [ "$DRV_MAJOR" -ge 580 ] || fail "torch is cu$TORCH_CUDA but the driver is R$DRV_MAJOR; R580 or newer is required" ;;
  12.*) [ "$DRV_MAJOR" -ge 525 ] || fail "torch is cu$TORCH_CUDA but the driver is R$DRV_MAJOR" ;;
  *)    fail "could not read a CUDA build from torch" ;;
esac

step "3. CUDA tensor, one operation, synchronize"
"$PY" - <<'PYEOF' || fail "CUDA is not usable on this host"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
torch.cuda.init()
dev = torch.device("cuda:0")
print("  device:", torch.cuda.get_device_name(0))
a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
b = (a @ a).float().sum()
torch.cuda.synchronize()
assert torch.isfinite(b).item(), "matmul produced a non-finite result"
free, total = torch.cuda.mem_get_info()
cc = torch.cuda.get_device_capability(0)
print(f"  matmul ok, sum={b.item():.3e}")
print(f"  compute capability = {cc[0]}.{cc[1]}")
print(f"  memory free/total = {free/2**30:.1f}/{total/2**30:.1f} GiB")
# Optional gates. Unset means "record it"; set means "this run is pinned to that card
# and has to fail loudly if it landed somewhere else". Which card the grid was fixed
# on is a property the results must carry, not something to infer afterwards.
import os
want_cc = os.environ.get("STAGE0_EXPECT_CC")
if want_cc and want_cc != f"{cc[0]}.{cc[1]}":
    raise SystemExit(f"compute capability is {cc[0]}.{cc[1]}, expected {want_cc}")
want_vram = os.environ.get("STAGE0_EXPECT_VRAM_GIB")
if want_vram and abs(total / 2**30 - float(want_vram)) > 2.0:
    raise SystemExit(f"total VRAM is {total/2**30:.1f} GiB, expected about {want_vram}")
PYEOF

step "4. vLLM and its compiled extension"
cd "$VLLM_DIR"
TREE=$(git write-tree)
echo "  tree       = $TREE"
echo "  expected   = $EXPECTED_TREE"
[ "$TREE" = "$EXPECTED_TREE" ] || fail "engine tree is not main + the BS x ctx patch"
[ "$(git rev-parse HEAD)" = "$EXPECTED_HEAD" ] || fail "engine base is not $EXPECTED_HEAD"
git diff --quiet || fail "unstaged changes on top of the patched index"
echo "  head       = $(git rev-parse HEAD)"
"$PY" - <<'PYEOF' | tee "$OUT/vllm_load.txt" || fail "vLLM or its compiled extension failed to load"
import importlib, json, torch, vllm
out = {"vllm_version": vllm.__version__, "vllm_file": vllm.__file__}
# The precompiled wheel's binaries are built from some upstream base commit, not
# from the patched tree. The patch touches Python only, so this is workable, but
# the provenance has to be recorded rather than assumed.
# This tree loads kernels through current_platform.import_kernels(), which
# vllm._custom_ops calls at import time. There is no importable vllm._C module
# here: the objects are named _C_stable_libtorch.abi3.so and friends, so asking
# for vllm._C would fail on a perfectly good host.
import glob, os
import vllm._custom_ops as vllm_ops
out["extension_objects"] = sorted(
    os.path.basename(f)
    for f in glob.glob(os.path.join(os.path.dirname(vllm.__file__), "*.so")))
out["custom_ops_present"] = hasattr(torch.ops, "_C")
out["rms_norm_registered"] = hasattr(torch.ops._C, "rms_norm")
out["compute_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
# Importing the extension only proves the shared object loads. The wheel was built
# from an upstream base commit for some set of architectures, and a kernel launch can
# still fail after a clean import, so run one compiled op on the device before
# anything large is downloaded.
x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)
vllm_ops.rms_norm(y, x, w, 1e-6)
torch.cuda.synchronize()
assert torch.isfinite(y).all().item(), "rms_norm produced a non-finite result"
out["compiled_op_executed"] = "vllm._custom_ops.rms_norm on cuda:0"
try:
    out["vllm_commit_attr"] = getattr(vllm, "__commit__", None)
except Exception as e:
    out["vllm_commit_attr"] = f"unavailable: {e}"
print(json.dumps(out, indent=2))
PYEOF
"$PIP" freeze | grep -iE '^(torch|torchvision|torchaudio|vllm|transformers|flashinfer)' | tee "$OUT/pins.txt"
# Record where the precompiled binaries came from, so a later run can pin it.
grep -oE 'https?://[^ ]*\.whl' /root/install_vllm.log 2>/dev/null \
  | grep -i vllm | sort -u | tee "$OUT/precompiled_wheel_urls.txt" || true
echo "  precompiled wheel URLs recorded: $(wc -l < "$OUT/precompiled_wheel_urls.txt" 2>/dev/null || echo 0)"

step "4a. source and binaries actually loaded"
"$PY" "$SMOKE/box/check_import_paths.py" "$VLLM_DIR" "$OUT/wheel_provenance.json" \
  | tee "$OUT/import_paths.json" | tail -1 | grep -q IMPORT_PATHS_OK \
  || fail "a module or compiled object is not the patched tree's or not the pinned wheel's; see $OUT/import_paths.json"
echo "  modules from $VLLM_DIR, mapped objects identical to the pinned wheel"

step "4b. trace hook (scheduler + MRv2 graph managers)"
if "$PY" -c "import sitecustomize" 2>/dev/null; then
  fail "a sitecustomize already exists on the default path; the trace hook would be shadowed"
fi
MRV2_TRACE_DIR=$OUT/trace_check PYTHONPATH="$SMOKE/trace" "$PY" - "$SMOKE/trace" <<'PYEOF' \
  || fail "the trace hook did not install"
import os, sys, sitecustomize
import vllm.v1.core.sched.scheduler as m
import vllm.v1.worker.gpu.cudagraph_utils as g
import vllm.v1.engine.core as e
assert getattr(e.EngineCore.__dict__["_wait_for_output"], "_mrv2_traced", False), "EngineCore wait not wrapped"
assert getattr(e.EngineCoreProc.__dict__["_wait_for_output"], "_mrv2_traced", False), "EngineCoreProc wait not wrapped"
# Production EngineCoreProc._wait_for_output on a fixture instance and a done future: a wait row
# naming EngineCoreProc must reach the trace file.
import glob, json, queue, time
from concurrent.futures import Future
proc = object.__new__(e.EngineCoreProc)
proc.input_queue, proc._deferred_inputs = queue.Queue(), []
fut = Future(); fut.set_result("ok")
assert proc._wait_for_output(fut) == "ok"
import mrv2_trace; mrv2_trace.tracer().f.flush()
rows = [json.loads(l) for p in glob.glob(os.path.join(os.environ["MRV2_TRACE_DIR"], "mrv2_trace_*.jsonl")) for l in open(p)]
assert any(r["ev"] == "wait" and r.get("cls") == "EngineCoreProc" and r.get("impl") == "EngineCoreProc" for r in rows), "no EngineCoreProc wait row"
assert os.path.realpath(sitecustomize.__file__).startswith(os.path.realpath(sys.argv[1])), sitecustomize.__file__
assert m.Scheduler.__dict__.get("_mrv2_traced"), "Scheduler not wrapped"
assert g.CudaGraphManager.__dict__.get("_mrv2_init_traced"), "CudaGraphManager not wrapped"
assert m.dynamic_sd_context_length.__name__ == "traced_ctx", "context lookup not wrapped"
print("  trace hook installs on", m.__file__, "and", g.__file__, "; EngineCoreProc wait row written")
PYEOF

step "4c. pre-download gates"
# Everything here is cheap and fails before the 37 GB is paid for. The last run
# learned this the hard way in the other direction: a marker was printed after a
# failed install because nothing was checked.
MODELS_ROOT=/root
NEED_GIB=45
AVAIL_GIB=$(df -Pk "$MODELS_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
echo "  free space on $MODELS_ROOT = ${AVAIL_GIB} GiB, need ${NEED_GIB} GiB"
[ "$AVAIL_GIB" -ge "$NEED_GIB" ] \
  || fail "only ${AVAIL_GIB} GiB free on $MODELS_ROOT; the weights are 37.1 GB and the image and wheels are already on this disk"
# Both pinned repos are public: a ranged GET for the target weights returns 206
# without any Authorization header, and the HEAD reports x-linked-size
# 36086696440, the pinned byte count. So a token is optional here. If one is
# present it is used; if not, the pull is unauthenticated and that is recorded
# rather than guessed at later. The token is never printed either way.
if [ -s /root/.hf_token ]; then
  HF_AUTH=1
  echo "  HF token file present (contents never printed)"
else
  HF_AUTH=0
  echo "  no HF token file; the pinned repos are public, pulling unauthenticated"
fi
echo "$HF_AUTH" > "$OUT/hf_auth.txt"
# The config contract this engine tree needs, checked on the two config.json
# files alone before 37 GB is pulled. Attempt 2 reached this point with
# transformers 5.20.2 installed, which folds Gemma4's global_head_dim into
# per-layer overrides, and vllm serve died in get_head_size. Configs are a few
# KB and need no token, so this is cheap insurance.
CFG_DIR=$OUT/cfg
mkdir -p "$CFG_DIR/target" "$CFG_DIR/draft"
curl -sS -L --fail -o "$CFG_DIR/target/config.json" \
  "https://huggingface.co/prithivMLmods/gemma-4-31B-it-qat-FP8/resolve/$TARGET_REV/config.json" \
  || fail "could not fetch the target config.json"
curl -sS -L --fail -o "$CFG_DIR/draft/config.json" \
  "https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant/resolve/$DRAFT_REV/config.json" \
  || fail "could not fetch the draft config.json"
( cd "$SMOKE" && "$PY" -c 'import py_compile; [py_compile.compile(f, doraise=True) for f in ("analyze_smoke.py", "probe_tokens.py", "compare_tokens.py", "make_probe_prompts.py")]' ) \
  || fail "the smoke tools do not compile on this host"
bash -n "$SMOKE/run_smoke.sh" || fail "run_smoke.sh does not parse"
echo "  smoke tools compile"
"$PY" "$SMOKE/box/check_b1_configs.py" "$CFG_DIR/target" "$CFG_DIR/draft" \
  | tee "$OUT/config_contract.json" | tail -1 | grep -q CONFIG_CONTRACT_OK \
  || fail "the per-layer config contract does not hold with the installed transformers; see $OUT/config_contract.json"
echo "  per-layer config contract holds for target and draft"

if [ "${1:-}" != "--download" ]; then
  echo
  echo "PREFLIGHT_RUNTIME_OK (checks 1-4c). Re-run with --download to pull the models."
  exit 0
fi

step "5. models, only now"
umask 077
CURL_AUTH=()
if [ "${HF_AUTH:-0}" = 1 ]; then
  printf 'header = "Authorization: Bearer %s"\n' "$(cat /root/.hf_token)" > /root/.curlrc.hf
  trap 'rm -f /root/.curlrc.hf' EXIT
  CURL_AUTH=(--config /root/.curlrc.hf)
fi
mkdir -p /root/models/target /root/models/draft
date +%s > "$OUT/t_download_start"
get() {  # repo rev dest file
  curl -sS -L --fail --retry 3 "${CURL_AUTH[@]+"${CURL_AUTH[@]}"}" \
       -o "$3/$4" "https://huggingface.co/$1/resolve/$2/$4" \
    || fail "download failed: $1 $4"
}
for f in config.json generation_config.json tokenizer_config.json \
         chat_template.jinja processor_config.json recipe.yaml \
         tokenizer.json model.safetensors; do
  echo "  target/$f"
  get prithivMLmods/gemma-4-31B-it-qat-FP8 "$TARGET_REV" /root/models/target "$f"
done
for f in config.json generation_config.json tokenizer_config.json \
         chat_template.jinja tokenizer.json model.safetensors; do
  echo "  draft/$f"
  get google/gemma-4-31B-it-qat-q4_0-unquantized-assistant "$DRAFT_REV" /root/models/draft "$f"
done
date +%s > "$OUT/t_download_end"
rm -f /root/.curlrc.hf
du -sb /root/models/target /root/models/draft | tee "$OUT/model_bytes.txt"
# Exact sizes at the pinned revisions, read from the Hugging Face API.
TB=$(stat -c %s /root/models/target/model.safetensors)
DB=$(stat -c %s /root/models/draft/model.safetensors)
[ "$TB" -eq 36086696440 ] || fail "target weights are $TB bytes, expected 36086696440 at $TARGET_REV"
[ "$DB" -eq 939042560 ]   || fail "draft weights are $DB bytes, expected 939042560 at $DRAFT_REV"
echo "  weight sizes match the pinned revisions"

step "6. manifest"
# One file that pins what this run actually was. The attempt-1 record could not
# say which runtime it used, and attempt 2 could not say which checkpoint Surface
# A used; neither gap is acceptable twice.
"$PY" - > "$OUT/manifest.json" <<'PYEOF'
import json, os, subprocess, transformers, torch, vllm
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as exc:
        return f"<{exc}>"
print(json.dumps({
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "driver_cuda_runtime": sh("nvidia-smi | sed -n 's/.*CUDA Version: *\\([0-9.]*\\).*/\\1/p'"),
    "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm_version": vllm.__version__,
    "vllm_tree": sh("git -C /workspace/vllm write-tree"),
    "vllm_base": sh("git -C /workspace/vllm rev-parse HEAD"),
    "patch": "dc/bs-ctx-dsd-zfix bd09764c45 = 8e5f0678a1 + draft_decode capture fix + test updates (local, not pushed)",
    "wheel_sha256": open("/root/preflight/wheel_sha256.txt").read().split()[0]
        if os.path.exists("/root/preflight/wheel_sha256.txt") else None,
    "target_repo": "prithivMLmods/gemma-4-31B-it-qat-FP8",
    "target_revision": os.environ.get("TARGET_REV", ""),
    "draft_repo": "google/gemma-4-31B-it-qat-q4_0-unquantized-assistant",
    "draft_revision": os.environ.get("DRAFT_REV", ""),
    "target_weight_bytes": os.path.getsize("/root/models/target/model.safetensors")
        if os.path.exists("/root/models/target/model.safetensors") else None,
    "draft_weight_bytes": os.path.getsize("/root/models/draft/model.safetensors")
        if os.path.exists("/root/models/draft/model.safetensors") else None,
    "hf_authenticated": os.path.exists("/root/.hf_token"),
    "stage0_deadline": os.environ.get("STAGE0_DEADLINE", ""),
    "precompiled_wheel_urls": open("/root/preflight/precompiled_wheel_urls.txt").read().split()
        if os.path.exists("/root/preflight/precompiled_wheel_urls.txt") else [],
}, indent=2))
PYEOF
TARGET_REV=$TARGET_REV DRAFT_REV=$DRAFT_REV true
cat "$OUT/manifest.json"

step "7. marker"
echo "all required checks passed"
date +%s > "$OUT/t_preflight_done"
echo "PREFLIGHT_OK"
