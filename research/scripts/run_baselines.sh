#!/usr/bin/env bash
# Week-1 baselines: start vLLM with each offload config, replay the trace,
# collect results/<run>/<config>/{requests.csv,metrics_delta.json,server.log}.
#
# Usage (from the vllm repo root, on the GPU box):
#   TRACE=traces/syn.jsonl FS_DIR=/mnt/nvme/kvplace research/scripts/run_baselines.sh
#   CONFIGS="cpu_lru tier_lru" research/scripts/run_baselines.sh   # subset
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TRACE=${TRACE:?set TRACE to a trace JSONL}
FS_DIR=${FS_DIR:-/tmp/kvplace_fs}
PORT=${PORT:-8000}
GPU_BLOCKS=${GPU_BLOCKS:-3000}          # forces GPU prefix-cache pressure
CPU_BYTES=${CPU_BYTES:-10000000000}     # total across workers
BLOCK_SIZE=${BLOCK_SIZE:-16}
MAX_LEN=${MAX_LEN:-32768}
TIME_SCALE=${TIME_SCALE:-1.0}
PREDICTOR=${PREDICTOR:-tool}            # same hint-only traffic for all configs
CONFIGS=${CONFIGS:-"gpu_only cpu_lru cpu_arc tier_lru tier_arc"}
RUN=${RUN:-$(date +%Y%m%d_%H%M%S)}
OUT=research/results/$RUN
PY=${PY:-.venv/bin/python}

export PYTHONPATH="$PWD/research${PYTHONPATH:+:$PYTHONPATH}"

kv_config() {
  local policy=$1 tiers=$2
  local extra="\"block_size\": $BLOCK_SIZE, \"cpu_bytes_to_use\": $CPU_BYTES"
  case $policy in
    lru|arc) extra+=", \"eviction_policy\": \"$policy\"" ;;
    reuse) extra+=", \"eviction_policy\": \"ReuseAwareCachePolicy\""
           extra+=", \"cache_policy_module_path\": \"kvplace.vllm_policy\"" ;;
  esac
  if [[ $tiers == tier ]]; then
    extra+=", \"spec_name\": \"TieringOffloadingSpec\""
    extra+=", \"secondary_tiers\": [{\"type\": \"fs\", \"root_dir\": \"$FS_DIR\"}]"
  fi
  echo "{\"kv_connector\": \"OffloadingConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {$extra}}"
}

# Sets EXTRA_ARGS for a config name.
set_args() {
  EXTRA_ARGS=()
  case $1 in
    gpu_only) return ;;
    cpu_lru)    EXTRA_ARGS=(--kv-transfer-config "$(kv_config lru cpu)") ;;
    cpu_arc)    EXTRA_ARGS=(--kv-transfer-config "$(kv_config arc cpu)") ;;
    cpu_reuse)  EXTRA_ARGS=(--kv-transfer-config "$(kv_config reuse cpu)") ;;
    tier_lru)   EXTRA_ARGS=(--kv-transfer-config "$(kv_config lru tier)") ;;
    tier_arc)   EXTRA_ARGS=(--kv-transfer-config "$(kv_config arc tier)") ;;
    tier_reuse) EXTRA_ARGS=(--kv-transfer-config "$(kv_config reuse tier)") ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

wait_healthy() {
  for _ in $(seq 600); do
    curl -sf "localhost:$PORT/health" >/dev/null && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died" >&2; return 1; }
    sleep 1
  done
  return 1
}

# FS_DIR is wiped between configs; refuse anything that doesn't look scratch.
[[ $FS_DIR == *kvplace* ]] || { echo "FS_DIR must contain 'kvplace'" >&2; exit 1; }

mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/vllm_commit.txt"
cp "$TRACE" "$OUT/trace.jsonl"

for cfg in $CONFIGS; do
  echo "=== $cfg ==="
  dir=$OUT/$cfg
  mkdir -p "$dir"
  rm -rf "$FS_DIR" && mkdir -p "$FS_DIR"
  set_args "$cfg"
  .venv/bin/vllm serve "$MODEL" --port "$PORT" \
    --block-size "$BLOCK_SIZE" --max-model-len "$MAX_LEN" \
    --num-gpu-blocks-override "$GPU_BLOCKS" \
    --enable-prompt-tokens-details \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "$dir/server.log" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
  wait_healthy
  $PY -m kvplace.replay "$TRACE" --url "http://localhost:$PORT" \
    --predictor "$PREDICTOR" --time-scale "$TIME_SCALE" --out "$dir" \
    --block-size "$BLOCK_SIZE" \
    | tee "$dir/summary.txt"
  kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
done
echo "results in $OUT"
