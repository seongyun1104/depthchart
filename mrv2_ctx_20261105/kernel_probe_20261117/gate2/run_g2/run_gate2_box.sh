#!/usr/bin/env bash
# Gate 2 box script (BUILD_PLAN.md): precompiled wheel of 7ecc83a9 as in run 2, then the built
# _C_stable_libtorch (main + sm90_fp8_candidates.patch) swapped in. Per arm, one process each:
# identity (stop on any mismatch) -> correctness (recorded, no stop) -> timing, two passes in mirrored order
# -> ncu on gate_up after a one-kernel filter probe. Marker in /root/costmap_out: ALL_DONE or STOPPED.
#   bash /root/kp/run_gate2_box.sh
# Local rehearsal overrides paths and skips provisioning: see rehearse_local.sh.
set -euo pipefail
OUT=${OUT:-/root/costmap_out}; KP=${KP:-/root/kp}; PY=${PY:-/venv/main/bin/python}; G2=${G2:-/root/g2}
SO=${SO:-/workspace/vllm/vllm/_C_stable_libtorch.abi3.so}; BENCH=${BENCH:-/workspace/vllm/benchmarks/kernels}
mkdir -p "$OUT" "$G2"
step=start
trap 'rc=$?; [ -e "$OUT/ALL_DONE" ] || [ -e "$OUT/STOPPED" ] || printf "BOX_STEP_FAILED %s rc=%s\n" "$step" "$rc" > "$OUT/STOPPED"' EXIT
log() { echo "$* $(date -u +%H:%M:%SZ)" | tee -a "$OUT/box.log"; }
echo "PACKET_SHA256 $(cat $KP/PACKET_SHA256)" | tee "$OUT/box.log"
step=provision; log PROVISION_START
if [ "${REHEARSAL:-0}" != 1 ]; then
  PATCH=/root/smoke/box/bs_ctx_bd09764c45.patch TRANSFORMERS_PIN=5.22.1 PANDAS_PIN=3.1.1 \
    bash /root/smoke/box/provision_mrv2.sh >> "$OUT/box.log" 2>&1
  # triton.testing's perf_report imports pandas and matplotlib unconditionally; neither is a vLLM dependency
  step=harness_deps
  /venv/main/bin/pip install -q matplotlib==3.11.2 >> "$OUT/box.log" 2>&1
fi
step=harness_imports
$PY -c "import matplotlib, pandas, triton.testing; from vllm._custom_ops import cutlass_scaled_mm; print('HARNESS_IMPORTS_OK', matplotlib.__version__, pandas.__version__)" | tee -a "$OUT/box.log"
cp /root/preflight/pins.txt /root/preflight/wheel_sha256.txt "$OUT/" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv > "$OUT/gpu.csv"
env | grep -E '^VLLM_' > "$OUT/vllm_env.txt" || true

step=so_files
WHEEL_SHA=$(sha256sum "$SO" | cut -c1-64)
CAND_SHA=$(cat $KP/CAND_SO_SHA256)
[ "$(sha256sum $KP/cand_C_stable_libtorch.abi3.so | cut -c1-64)" = "$CAND_SHA" ]
[ "$WHEEL_SHA" != "$CAND_SHA" ]
cp "$SO" "$G2/wheel_C_stable_libtorch.abi3.so"
echo "wheel $WHEEL_SHA" > "$OUT/so_sha256.txt"; echo "cand $CAND_SHA" >> "$OUT/so_sha256.txt"
use_so() {  # W -> wheel's .so, else the built one; the swap is checked by hash every time
  if [ "$1" = W ]; then cp "$G2/wheel_C_stable_libtorch.abi3.so" "$SO"; want=$WHEEL_SHA
  else cp $KP/cand_C_stable_libtorch.abi3.so "$SO"; want=$CAND_SHA; fi
  [ "$(sha256sum "$SO" | cut -c1-64)" = "$want" ] || { echo "SO_SWAP_MISMATCH $1" | tee -a "$OUT/box.log"; return 1; }
}
arm_env() {  # prints the env assignments for one arm; selectors always cleared first
  case $1 in
    W|0) echo "" ;;
    1|2|3|4) echo "VLLM_TEST_SM90_FP8_M128_CAND=$1" ;;
    P) echo "VLLM_TEST_SM90_FP8_PR66852=1" ;;
  esac
}
run_arm() {  # run_arm <arm> <cmd...>
  local arm=$1; shift
  use_so "$arm" || return 1
  # shellcheck disable=SC2046
  env -u VLLM_TEST_SM90_FP8_M128_CAND -u VLLM_TEST_SM90_FP8_PR66852 -u VLLM_BATCH_INVARIANT \
    G2_EXPECT_SO_SHA="$want" $(arm_env "$arm") "$@"
}
ARMS="W 0 1 2 3 4 P"

step=identity; log IDENTITY_START
for arm in $ARMS; do
  step=identity_$arm
  run_arm "$arm" $PY $KP/gate2_check.py --arm "$arm" --mode identity > "$OUT/identity_$arm.jsonl" 2> "$OUT/identity_$arm.err"
  log "IDENTITY_OK $arm"
done

step=correct; log CORRECT_START
for arm in 0 1 2 3 4 P W; do  # arm 0 first: it writes the reference outputs
  set +e
  run_arm "$arm" $PY $KP/gate2_check.py --arm "$arm" --mode correct --ref "$G2/ref" > "$OUT/correct_$arm.jsonl" 2> "$OUT/correct_$arm.err"
  rc=$?; set -e
  log "CORRECT $arm rc=$rc $(tail -1 "$OUT/correct_$arm.jsonl")"
  [ "$arm" != 0 ] || [ "$rc" = 0 ] || { step=correct_arm0_failed; exit 1; }
done

step=timing; log TIMING_START
cd "$BENCH"
patch -p1 < $KP/harness_gate2.patch >> "$OUT/box.log"
MODELS="gate2/gemma-4-31B-fp8 gate2/control-n2048"
for pass in 1 2; do
  order=$ARMS; [ $pass = 2 ] && order="P 4 3 2 1 0 W"
  for arm in $order; do
    step=timing_p${pass}_$arm
    rm -rf bench_fp8_res_*
    $PY $KP/make_result_dirs.py $MODELS >> "$OUT/box.log"
    run_arm "$arm" env MPLBACKEND=Agg $PY benchmark_fp8_gemm.py --models $MODELS > "$OUT/bench_p${pass}_$arm.log" 2>&1
    mkdir -p "$OUT/time/p${pass}_$arm" && cp -r bench_fp8_res_* "$OUT/time/p${pass}_$arm/"
    log "TIMING_DONE p$pass $arm"
  done
done
cd "$G2"

step=ncu
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1 || true)
if [ -n "$NCU" ]; then
  set +e
  # gate 1 filtered on "cutlass" and matched nothing: ncu matches the base name device_kernel. Probe first.
  run_arm 0 "$NCU" --kernel-name regex:device_kernel --launch-count 1 --metrics gpu__time_duration.sum --csv \
    $PY $KP/gate2_check.py --arm 0 --mode ncu > "$OUT/ncu_probe.csv" 2> "$OUT/ncu_probe.err"
  rc=$?
  n=$(grep -c '^".*device_kernel' "$OUT/ncu_probe.csv")
  log "NCU_PROBE rc=$rc kernels=$n $NCU"
  if [ "$n" = 1 ]; then
    for arm in 0 1 2 3 4; do
      run_arm "$arm" "$NCU" --kernel-name regex:device_kernel --metrics dram__bytes_read.sum,gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__cluster_size --csv \
        $PY $KP/gate2_check.py --arm "$arm" --mode ncu > "$OUT/ncu_a_$arm.csv" 2> "$OUT/ncu_a_$arm.err"
      log "NCU_A $arm rc=$? kernels=$(grep -c '^".*device_kernel' "$OUT/ncu_a_$arm.csv")"
      run_arm "$arm" "$NCU" --kernel-name regex:device_kernel --metrics sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active --csv \
        $PY $KP/gate2_check.py --arm "$arm" --mode ncu > "$OUT/ncu_b_$arm.csv" 2> "$OUT/ncu_b_$arm.err"
      log "NCU_B $arm rc=$? kernels=$(grep -c '^".*device_kernel' "$OUT/ncu_b_$arm.csv")"
    done
  fi
  set -e
else
  log "NCU absent"
fi
touch "$OUT/ALL_DONE"
