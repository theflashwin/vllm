# GPU admission changes the results

**Summary.** The simulator treated GPU blocks as a cache only: a running
request used no GPU memory. In vLLM, a running request holds its whole
context in GPU blocks until it finishes. Modeling that (`--gpu-admission`)
reproduces the A10G live run, and shows that the headline settings were
overloaded. At loads a server can sustain, we found no setting on this trace
where CPU-tier eviction by predicted return clearly beats LRU.

## The admission model

`Simulator(gpu_admission=True)` / `kvplace.sim --gpu-admission`: each turn
and hint-only request holds `ceil((prompt + output) / 16)` GPU blocks from
start to finish, and starts only when that many blocks are free (FCFS, like
vLLM's waiting queue). Off by default, so earlier results reproduce.

## It matches live

A10G, `live70` (token scale 0.15, 3800 GPU blocks, 3.67 GB CPU tier, time
scale 0.2), CPU tier LRU:

| | p50 | p90 | p99 | Wall |
|---|---:|---:|---:|---:|
| Live | 14.7 s | 44.9 s | 60.9 s | 55 min |
| Simulator | 0.06 s | 0.87 s | 2.3 s | ~30 min |
| Simulator + admission | 15.8 s | 31.3 s | 40.9 s | ~45 min |

The remaining p90/p99 gap is consistent with disk thrash (live read 3.54 TB
from disk, ~60x the unique KV), which is not modeled yet.

## The headline settings were overloaded

Sample 1, `tool-nfh`, p50 / p90 resume TTFT:

| Setting | Policy | No admission | Admission |
|---|---|---:|---:|
| 12k GPU / 96k CPU, 3 GB/s | GPU-aware | 0.32 / 2.52 s | 1271 / 1869 s |
| | Ours | 0.32 / 1.81 s | 1209 / 1912 s |
| L4: 8k GPU / 48k CPU | GPU-aware | 0.91 / 14.1 s | 4570 / 7472 s |
| | Ours | 0.83 / 7.32 s | 4488 / 7227 s |

Median contexts are ~80k tokens (~5000 blocks), so these GPUs run only 1-2
requests at a time and requests queue for tens of minutes. The earlier gains
(-21% to -61%) were measured in a regime no real server would run.

## At sustainable load, no clear gain

Single samples, `tool-nfh2`, contexts capped at 128k, real waits,
`--gpu-admission --contend-sec --write-model shared`. CPU tier LRU vs ours:

| Setting | LRU p90 / p99 | Ours p90 / p99 | Resumes from disk |
|---|---:|---:|---:|
| A100-80 (60k blocks), 0.02 sessions/s, 7 GB CPU | 0.61 / 4.86 s | 0.60 / 4.86 s | 1.6% |
| A100-80, 0.02/s, 29 GB CPU | 0.60 / 4.80 s | 0.60 / 4.80 s | 1.6% |
| A100-40 (22k blocks), 0.01/s, 7 GB CPU | 3.51 / 29.0 s | 3.47 / 26.7 s | 4.3% |
| A100-40, 0.01/s, 29 GB CPU | 1.50 / 16.7 s | 1.60 / 15.6 s | 2.2-2.4% |
| A100-40, 0.005/s, 7 GB CPU | 0.61 / 7.19 s | 0.61 / 7.28 s | 2.4% |
| A100-40, 0.005/s, 29 GB CPU | 0.56 / 4.80 s | 0.57 / 4.72 s | 1.2-1.5% |

A100-80 at 0.05 sessions/s is overloaded (queue p90 ~800-900 s). On the A10G
(35-50 sessions of `live70`, 1-2 GB CPU tier, time scale 0.2 or 0.5) the
p90 change ranges from -3% to +14% and p99 from -24% to +18%, with no
consistent direction.

At a sustainable load the GPU cache already holds most prefixes that come
back soon; only 1-7% of resumes load from disk, so the CPU tier's eviction
order rarely matters.

## Caveats and next steps

- Single samples; repeat the stable settings over 5 samples for confidence
  intervals.
- Not modeled yet: disk thrash (staged reads need CPU space), batched decode
  slowdown.
- One trace. The open question is whether any realistic workload has idle KV
  far beyond CPU memory at a sustainable load: longer retention, larger KV
  per token (70B), or more idle agents per GPU (CATraces).
- Earlier simulator results (DIRECTION.md, the midpoint slides, the paired100
  A100 prediction) were all run without admission and should be read with
  this in mind.

## Reproduce

```bash
cd research
PYTHONPATH=. python -m kvplace.tracelab traces/syfi_coding_trace.jsonl.gz \
  -o traces/feas_r0.02.jsonl --provider claude --rebase-rate 0.02 \
  --sessions 300 --max-context 131072 --seed 1
PYTHONPATH=. python -m kvplace.sim traces/feas_r0.02.jsonl --gpu-blocks 60000 \
  --cpu-blocks 8000 --policies lru reuse_evict --predictors tool-nfh2 \
  --cost-json costs_a100.json --contend-sec --write-model shared --gpu-admission
```
