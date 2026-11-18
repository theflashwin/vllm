# PACE runs to queue, in priority order (updated Nov 17)

Run on PACE from `/storage/ice1/6/7/jmo46/vllm` at `41c463703d` or later
(tests pass there). Submit in this order; if the queue is slow, the first steps
answer the main question.

Done: live70 load check (7124171, 7124172) and repeat 1 of LRU and ours
(`tool-nfh2`) at 12.6 and 21 GB, disk 0.25 GB/s (7124913-7124916).

"Ours" is now `tool-nfhc`: `tool-nfh2`-style prediction plus compaction and
context-limit ends flagged final, as on the simulator slide. LRU ignores hints,
so its predictor does not matter.

## Variables (every new terminal)

```bash
TS=2.0; T=4:30:00; DL=14400; L70=research/traces/live/live70_ts0.35.jsonl
```

## 1. live70, repeat 1: ours and true reuse times (4 jobs)

Completes repeat 1 next to the finished LRU runs. Simulator: compaction hints
-10% p99 at 21 GB and -20% at 12.6 GB; true reuse times -78% at 21 GB and -31%
at 12.6 GB.

```bash
for cpu in 12582912000 20971520000; do for pred in tool-nfhc oracle; do TRACE=$L70 CONFIGS=tier_reuse PREDICTOR=$pred CPU_BYTES=$cpu DISK_GB_S=0.25 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e1 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done
```

## 2. Shared disk, live70 at 21 GB (3 jobs)

Reads and writes share one 0.25 GB/s device, so write-through traffic delays
resume loads. Largest simulated gap: LRU p99 9.2 s, true reuse times -80%,
compaction hints -1%.

```bash
for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfhc tier_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=20971520000 DISK_GB_S=0.25 DISK_SHARED=1 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e5 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done
```

## 3. Longer contexts (3 jobs)

`tl70_longctx`: 70 public TraceLab sessions, contexts 2x longer (64k cap), GPU
cache doubled (7,600 blocks), CPU tier 21 GB, real disk. Simulator: ours -35%,
as good as true reuse times. The 4 h deadline covers about 75% of the trace,
the same window for every arm.

The trace is gitignored; copy it first, on the Mac:

```bash
rsync -av /Users/home/Documents/GitHub/vllm/research/traces/tl70/ jmo46@login-ice.pace.gatech.edu:/storage/ice1/6/7/jmo46/vllm/research/traces/tl70/
```

Then on PACE:

```bash
for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfhc tier_reuse:oracle; do TRACE=research/traces/tl70/tl70_longctx.jsonl CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=20971520000 GPU_BLOCKS=7600 MAX_LEN=65536 TIME_SCALE=2.0 DEADLINE_S=14400 sbatch -J kvplace-e3 --gres=gpu:l40s:1 --mem=64G -t 4:45:00 research/scripts/pace_live.sbatch; done
```

## 4. No disk tier (6 jobs)

Evicted KV must be recomputed, so a wrong eviction costs a full prefill. Same
arms and sizes as step 1, CPU tier only.

```bash
for cpu in 12582912000 20971520000; do for arm in cpu_lru:tool-nfh2 cpu_reuse:tool-nfhc cpu_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=$cpu TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e2 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done
```

## 5. Repeats 2 and 3 of steps 1-4 (36 jobs, queued Nov 17)

Steps 1-4 finished Nov 17 (one run each). These repeat every arm twice so the
paper can report run-to-run spread; the 12.6 GB gain needs them most. The old
`tool-nfh2` "ours" arm is not repeated.

```bash
for i in 2 3; do for cpu in 12582912000 20971520000; do for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfhc tier_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=$cpu DISK_GB_S=0.25 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e1 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done; done
for i in 2 3; do for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfhc tier_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=20971520000 DISK_GB_S=0.25 DISK_SHARED=1 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e5 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done
for i in 2 3; do for cpu in 12582912000 20971520000; do for arm in cpu_lru:tool-nfh2 cpu_reuse:tool-nfhc cpu_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=$cpu TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e2 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done; done
for i in 2 3; do for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfhc tier_reuse:oracle; do TRACE=research/traces/tl70/tl70_longctx.jsonl CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=20971520000 GPU_BLOCKS=7600 MAX_LEN=65536 TIME_SCALE=2.0 DEADLINE_S=14400 sbatch -J kvplace-e3 --gres=gpu:l40s:1 --mem=64G -t 4:45:00 research/scripts/pace_live.sbatch; done; done
```

## Checks on finished runs

`tool-nfhc` arms: the policy should read final hints by UDP.

```bash
grep -c "final=True.*(udp)" research/results/live/<run>/tier_reuse/server.log
```

Shared-disk runs: `node.txt` says "reads and writes shared" and `args.json`
has `"disk_shared": true`.

## Managing jobs

```bash
squeue -u $USER -o "%.10i %.13j %.8T %.10M %R"
scancel -u $USER -n kvplace-e2
```

Results land in `research/results/live/<trace>_<jobid>/`; `args.json` records
`cpu_bytes`, `predictor`, `disk_gb_s`, `disk_shared`, `gpu_blocks`, `max_len`
and the config. Copy them down with:

```bash
rsync -av jmo46@login-ice.pace.gatech.edu:/storage/ice1/6/7/jmo46/vllm/research/results/live/ /Users/home/Documents/GitHub/vllm/research/results/live/
```
