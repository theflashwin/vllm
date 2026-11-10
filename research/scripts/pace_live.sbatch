#!/bin/bash
#SBATCH --job-name=kvplace-live
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=kvplace-live-%j.out
#
# Live replay on a Slurm GPU node (PACE-ICE): the arms of modal_live.py, run
# through scripts/run_baselines.sh. Results go to research/results/live/<run>/.
#
# The partition and GPU names above are unconfirmed. Check them with
#   sinfo -o "%P %G %l %m"
# and override on the command line, which takes precedence over #SBATCH lines.
#
# One-time setup, from the vllm repo root in scratch (same vLLM as modal_live.py):
#   uv venv --python 3.12 && uv pip install vllm==0.31.0 aiohttp
#   export HF_HOME=<scratch>/hf    # in your shell profile; home quotas are small
#
# Usage, from the repo root (sbatch passes the environment through):
#   TRACE=research/traces/live/live35.jsonl sbatch research/scripts/pace_live.sbatch
#   TRACE=... CONFIGS="tier_reuse tier_lru" RUN=live35_b \
#     sbatch --gres=gpu:A100:1 --mem=96G research/scripts/pace_live.sbatch
# Check a node without starting a server:
#   TRACE=... DRY_RUN=1 srun --gres=gpu:1 --mem=64G bash research/scripts/pace_live.sbatch
#
# Defaults match results/live/live35_stable_a10g except for the hint transport
# (add HINT_PORT= to reproduce it exactly). Arms run in CONFIGS order;
# swap the order between repeats. results/ is gitignored: commit with `git add -f`.
# The replay writes its rows only when an arm ends, so a job killed at the
# walltime loses that arm. Set DEADLINE_S (per arm) so all arms fit in --time.
set -euo pipefail

die() { echo "pace_live: $*" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -x research/scripts/run_baselines.sh ]] || die "submit from the vllm repo root"

job=${SLURM_JOB_ID:-$$}
export TRACE=${TRACE:?set TRACE to a trace JSONL}
export CONFIGS=${CONFIGS:-"tier_lru tier_reuse"}
export PREDICTOR=${PREDICTOR:-tool-nfh2}
export TIME_SCALE=${TIME_SCALE:-0.5}
export GPU_BLOCKS=${GPU_BLOCKS:-3800}
export CPU_BYTES=${CPU_BYTES:-1835000000}
export RUN=${RUN:-$(basename "$TRACE" .jsonl)_$job}
export OUT=${OUT:-research/results/live/$RUN}
export FS_DIR=${FS_DIR:-${TMPDIR:-/tmp}/kvplace_fs_$job}
# GPU nodes are shared, so a fixed port can collide with another job.
export PORT=${PORT:-$((20000 + job % 20000))}
# Post-response hints go to the policy by UDP. HINT_PORT= (empty) sends them
# as hint-only requests through the engine instead, as runs before Nov 10 did.
export HINT_PORT=${HINT_PORT-$((PORT + 1))}
export HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1800}
export KVPLACE_LOG_HINTS=1
VENV=${VENV:-$PWD/.venv}
export PY=$VENV/bin/python VLLM=$VENV/bin/vllm

[[ -f $TRACE ]] || die "no trace at $TRACE"
[[ -x $VLLM ]] || die "no vllm in $VENV; see the setup lines in this script"
[[ -n ${HF_HOME:-} ]] || echo "pace_live: HF_HOME is unset; weights go to ~/.cache" >&2

# The CPU tier is DRAM inside the job's memory limit. The headroom for vLLM
# itself is an estimate.
if [[ -n ${SLURM_MEM_PER_NODE:-} ]]; then
  need_mb=$((CPU_BYTES / 1000000 + ${MEM_HEADROOM_MB:-16000}))
  ((need_mb <= SLURM_MEM_PER_NODE)) ||
    die "CPU tier needs about ${need_mb} MB, job has ${SLURM_MEM_PER_NODE} MB; raise --mem"
fi

mkdir -p "$FS_DIR" "$OUT"
trap 'rm -rf "$FS_DIR"' EXIT

# The disk tier must be a real disk with O_DIRECT. A RAM-backed /tmp would
# count against job memory and read back at DRAM speed.
fstype=$(stat -f -c %T "$FS_DIR")
case $fstype in
  tmpfs | ramfs) die "$FS_DIR is $fstype (RAM); set FS_DIR to a local disk" ;;
esac
probe=$FS_DIR/probe
wrote=$(dd if=/dev/zero of="$probe" bs=1M count=256 oflag=direct 2>&1 | tail -1) ||
  die "$FS_DIR ($fstype) does not support O_DIRECT; set FS_DIR to a local disk"
read_back=$(dd if="$probe" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1)
rm -f "$probe"

{
  echo "host: $(hostname)  job: $job  commit: $(git rev-parse HEAD)"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -4)"
  echo "mem: ${SLURM_MEM_PER_NODE:-?} MB  cpus: ${SLURM_CPUS_PER_TASK:-?}"
  echo "fs_dir: $FS_DIR ($fstype), $(df -h "$FS_DIR" | tail -1)"
  echo "direct write, 256 MiB of zeros (rough): $wrote"
  echo "direct read: $read_back"
  echo "configs: $CONFIGS  predictor: $PREDICTOR  time_scale: $TIME_SCALE"
  echo "gpu_blocks: $GPU_BLOCKS  cpu_bytes: $CPU_BYTES  port: $PORT"
  echo "hint transport: ${HINT_PORT:+udp port }${HINT_PORT:-request}"
} | tee "$OUT/node.txt"

if [[ -n ${DRY_RUN:-} ]]; then
  echo "pace_live: dry run, not starting a server"
  exit 0
fi

research/scripts/run_baselines.sh
