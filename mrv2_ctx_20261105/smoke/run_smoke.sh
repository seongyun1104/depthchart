#!/bin/bash
# MRv2 smoke on one H100 NVL. Not run yet. Each launch is its own server; a
# launch starts only if its estimate fits before DEADLINE_EPOCH. Stops (exit 3)
# when target FULL replay is not observed — no switch to PIECEWISE.
#
#   DEADLINE_EPOCH=<epoch> MODEL_DIR=... DRAFT_DIR=... SMOKE=<this dir> OUT=/root/smoke \
#     bash run_smoke.sh
set -euo pipefail
: "${DEADLINE_EPOCH:?}" "${MODEL_DIR:?}" "${DRAFT_DIR:?}" "${SMOKE:?}" "${OUT:?}"
PORT=8000
EST_LAUNCH_S=${EST_LAUNCH_S:-1500}
PROBE="$OUT/probe_prompts.jsonl"
mkdir -p "$OUT"
[ -s "$PROBE" ] || python "$SMOKE/make_probe_prompts.py" "$MODEL_DIR" "$PROBE"

draft_cfg() { python -c "import json,sys; c=json.loads(sys.argv[1]); c['model']=sys.argv[2]; print(json.dumps(c))" "$1" "$DRAFT_DIR"; }

fits() {
  local now; now=$(date +%s)
  [ $((now + EST_LAUNCH_S)) -lt "$DEADLINE_EPOCH" ] || { echo "DEADLINE: skip $1 (now=$now est=$EST_LAUNCH_S)"; return 1; }
}

wait_ready() {
  local pid=$1 t0=$2
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "server exited"; return 1; }
    [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || { echo "deadline while starting"; return 1; }
    sleep 5
  done
  echo $(( $(date +%s) - t0 ))
}

bench_cell() {  # load prefix name
  vllm bench serve --model "$MODEL_DIR" --port "$PORT" --dataset-name random \
    --random-input-len "$2" --random-output-len 256 --ignore-eos \
    --num-prompts "$1" --max-concurrency "$1" --seed 7 \
    --save-result --result-dir "$3" --result-filename "cell_L$1_P$2.json" \
    > "$3/cell_L$1_P$2.log" 2>&1
}

launch() {  # name spec_json(or "") ctx_arm(0/1) bench(0/1) probes
  local name=$1 spec=$2 ctx=$3 bench=$4 probes=$5 dir="$OUT/$1"
  fits "$name" || return 2
  mkdir -p "$dir/trace"
  local args=(serve "$MODEL_DIR" --port "$PORT" --gpu-memory-utilization 0.90
    --max-num-seqs 64 --max-model-len 34816)
  [ -n "$spec" ] && args+=(--speculative-config "$(draft_cfg "$spec")")
  local t0; t0=$(date +%s)
  MRV2_TRACE_DIR="$dir/trace" PYTHONPATH="$SMOKE/trace" VLLM_USE_V2_MODEL_RUNNER=1 \
    vllm "${args[@]}" > "$dir/server.log" 2>&1 &
  local pid=$!
  local ready; ready=$(wait_ready "$pid" "$t0") || { kill "$pid" || true; return 1; }
  echo "$name startup_s=$ready" | tee -a "$OUT/timing.txt"
  for i in $(seq 1 "$probes"); do
    local p0; p0=$(date +%s)
    python "$SMOKE/probe_tokens.py" "$PROBE" "$dir/tokens_$i.jsonl" --port "$PORT"
    echo "$name probe_${i}_s=$(( $(date +%s) - p0 ))" | tee -a "$OUT/timing.txt"
  done
  if [ "$bench" = 1 ]; then
    for load in 11 32; do for prefix in 4096 32768; do
      local b0; b0=$(date +%s)
      bench_cell "$load" "$prefix" "$dir"
      echo "$name cell_L${load}_P${prefix}_s=$(( $(date +%s) - b0 ))" | tee -a "$OUT/timing.txt"
    done; done
  fi
  kill "$pid"; wait "$pid" || true
  python "$SMOKE/analyze_smoke.py" "$dir/trace" $([ "$ctx" = 1 ] && echo --ctx-arm) > "$dir/gates.json"
  if [ -n "$spec" ]; then
    python - "$dir/gates.json" <<'EOF' || return 3
import json, sys
g = json.load(open(sys.argv[1]))
need = ("mrv2_used", "target_full_captured", "target_full_replayed")
bad = [k for k in need if not g[k]]
print("gates", {k: g[k] for k in need + ("drafter_prefill_full_replayed",)}, "drafter_decode", g["drafter_decode"])
sys.exit(3 if bad else 0)
EOF
  fi
}

S3='{"num_speculative_tokens": 3}'
X='{"num_speculative_tokens": 5, "num_speculative_tokens_per_batch_size_and_context": [[1, 16, 1, 8192, 3], [1, 16, 8193, 40000, 5], [17, 64, 1, 8192, 2], [17, 64, 8193, 40000, 3]]}'
B='{"num_speculative_tokens": 5, "num_speculative_tokens_per_batch_size": [[1, 16, 5], [17, 64, 3]]}'

launch L1_static_k3 "$S3" 0 0 1
launch L2_bs_ctx "$X" 1 1 1
launch L4_off "" 0 0 2
launch L3_bs_only "$B" 0 0 1 || echo "L3 not run (rc=$?)"
launch L5_off "" 0 0 1 || echo "L5 not run (rc=$?)"
python "$SMOKE/compare_tokens.py" "$OUT/L4_off/tokens_1.jsonl" "$OUT/L4_off/tokens_2.jsonl" > "$OUT/aa_within_launch.json"
[ -s "$OUT/L5_off/tokens_1.jsonl" ] && python "$SMOKE/compare_tokens.py" "$OUT/L4_off/tokens_1.jsonl" "$OUT/L5_off/tokens_1.jsonl" > "$OUT/aa_across_launch.json"
for arm in L1_static_k3 L2_bs_ctx L3_bs_only; do
  [ -s "$OUT/$arm/tokens_1.jsonl" ] || continue
  python "$SMOKE/compare_tokens.py" "$OUT/L4_off/tokens_1.jsonl" "$OUT/$arm/tokens_1.jsonl" > "$OUT/ab_${arm}_vs_off.json"
done
echo SMOKE_DONE
