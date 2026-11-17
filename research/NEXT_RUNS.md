# PACE runs to queue (updated Nov 17)

Run on PACE from `/storage/ice1/6/7/jmo46/vllm` unless noted. Steps 1-4 run
on the PACE checkout as it is (`da48000f58`). Steps 5-7 need the Nov 17 code
(the `tool-nfhc` predictor and `DISK_SHARED`): pull it first, see "Updating the
PACE checkout".

Already queued Nov 16: live70 load check (7124171, 7124172), the oracle smoke
test (passed: 55 of 55 request hints), and repeat 1 of LRU and ours at 12.6 and
21 GB (7124913-7124916, job name `kvplace-e1`).

## Variables (every new terminal)

```bash
TS=2.0; T=4:30:00; DL=14400; L70=research/traces/live/live70_ts0.35.jsonl
```

## 0. See what is already queued

```bash
sacct -u $USER -S 2026-11-16 -X --format=JobID,JobName%15,State,Elapsed | grep kvplace-e
```

## 1. Oracle arms, repeat 1 (2 jobs; skip if already queued)

```bash
for cpu in 12582912000 20971520000; do TRACE=$L70 CONFIGS=tier_reuse PREDICTOR=oracle CPU_BYTES=$cpu DISK_GB_S=0.25 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e1 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done
```

## 2. live70 mid-sized CPU tier, repeats 2 and 3 (12 jobs)

LRU, ours and ours with true reuse times at 12.6 and 21 GB, disk 0.25 GB/s.

```bash
for i in 2 3; do for cpu in 12582912000 20971520000; do for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfh2 tier_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=$cpu DISK_GB_S=0.25 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e1 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done; done
```

## 3. No disk tier, repeat 1 (6 jobs)

Evicted KV must be recomputed. Same arms and sizes, CPU tier only.

```bash
for cpu in 12582912000 20971520000; do for arm in cpu_lru:tool-nfh2 cpu_reuse:tool-nfh2 cpu_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=$cpu TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e2 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done
```

## 4. Longer contexts, repeat 1 (3 jobs)

`tl70_longctx`: 70 public TraceLab sessions, contexts 2x longer (64k cap). The
GPU cache is doubled with the contexts (7,600 blocks), CPU tier 21 GB, real
disk. The trace spans ~2.7 h at time scale 1.0; at 2.0 the 4 h deadline covers
about 75% of it, the same window for every arm.

First copy the trace, on the Mac (it is gitignored, so not on PACE):

```bash
rsync -av /Users/home/Documents/GitHub/vllm/research/traces/tl70/ jmo46@login-ice.pace.gatech.edu:/storage/ice1/6/7/jmo46/vllm/research/traces/tl70/
```

Then on PACE:

```bash
for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfh2 tier_reuse:oracle; do TRACE=research/traces/tl70/tl70_longctx.jsonl CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=20971520000 GPU_BLOCKS=7600 MAX_LEN=65536 TIME_SCALE=2.0 DEADLINE_S=14400 sbatch -J kvplace-e3 --gres=gpu:l40s:1 --mem=64G -t 4:45:00 research/scripts/pace_live.sbatch; done
```

## Updating the PACE checkout

The new code only adds options; every existing setting runs as before, so
jobs from steps 1-4 are unaffected. git replaces files rather than editing them
in place, so a running job keeps the version it started with.

```bash
git pull --ff-only && git log -1 --oneline
.venv/bin/python -m pytest research/tests/test_slow_disk.py research/tests/test_sim.py -q
```

## 5. Compaction hints, live70 (6 jobs)

`tool-nfhc` is `tool-nfh2` plus one signal the agent has: a segment that ends
in a compaction or a context-limit cut is flagged final (34 of 70 live70
segments). This is the "ours" of the simulator slide (live70: -10% p99 at 21
GB, -20% at 12.6 GB). LRU ignores hints, so the LRU and oracle runs from steps
1-2 are its baselines. Three repeats.

```bash
for i in 1 2 3; do for cpu in 12582912000 20971520000; do TRACE=$L70 CONFIGS=tier_reuse PREDICTOR=tool-nfhc CPU_BYTES=$cpu DISK_GB_S=0.25 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e4 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done; done
```

Check one finished arm: the policy should read final hints by UDP.

```bash
grep -c "final=True.*(udp)" research/results/live/live70_ts0.35_<jobid>/tier_reuse/server.log
```

## 6. Shared disk, live70 at 21 GB (3 jobs)

Reads and writes share one 0.25 GB/s device, so write-through traffic delays
resume loads (simulator: LRU p99 9.2 s; true reuse times -80%, compaction
hints -1%).

```bash
for arm in tier_lru:tool-nfh2 tier_reuse:tool-nfhc tier_reuse:oracle; do TRACE=$L70 CONFIGS=${arm%%:*} PREDICTOR=${arm#*:} CPU_BYTES=20971520000 DISK_GB_S=0.25 DISK_SHARED=1 TIME_SCALE=$TS DEADLINE_S=$DL sbatch -J kvplace-e5 --gres=gpu:l40s:1 --mem=64G -t $T research/scripts/pace_live.sbatch; done
```

`node.txt` should say "reads and writes shared" and `args.json` `"disk_shared": true`.

## 7. Compaction hints, longer contexts (1 job)

The `tool-nfhc` arm for step 4 (simulator: -35%, as good as true reuse times).
Needs the `tl70` trace copied over (step 4).

```bash
TRACE=research/traces/tl70/tl70_longctx.jsonl CONFIGS=tier_reuse PREDICTOR=tool-nfhc CPU_BYTES=20971520000 GPU_BLOCKS=7600 MAX_LEN=65536 TIME_SCALE=2.0 DEADLINE_S=14400 sbatch -J kvplace-e3 --gres=gpu:l40s:1 --mem=64G -t 4:45:00 research/scripts/pace_live.sbatch
```

## Managing jobs

```bash
squeue -u $USER -o "%.10i %.13j %.8T %.10M %R"
scancel -u $USER -n kvplace-e2
```

Results land in `research/results/live/<trace>_<jobid>/`; `args.json` records
`cpu_bytes`, `predictor`, `disk_gb_s`, `gpu_blocks`, `max_len` and the config.
Copy them down with:

```bash
rsync -av jmo46@login-ice.pace.gatech.edu:/storage/ice1/6/7/jmo46/vllm/research/results/live/ /Users/home/Documents/GitHub/vllm/research/results/live/
```
