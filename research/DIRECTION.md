# Project direction: agent-aware demotion to the disk tier

Status as of 2026-10-20. Simulator results over 5 trace samples, with one
setting calibrated by costs measured on a Modal L4 (`costs_l4.json`). No live
vLLM replay yet.

## Thesis

Agent-aware KV systems (CacheWise, TokenCake, MORI, EfficientAgent) decide
what stays on the GPU versus CPU. On hardware with an NVMe tier, the decision
that matters more is which sessions the CPU tier demotes to disk. Putting
agent awareness there, and *not* using it to prefetch from disk over a
contended link, reduces tail resumption latency beyond GPU/CPU-only
agent-aware policies.

**Hypothesis.** When the agents' KV working set exceeds GPU+CPU memory and
disk reads are a bottleneck, agent-aware CPU->disk demotion reduces p90
resumption TTFT by >=10% over a GPU/CPU-only agent-aware policy, with no
extra disk traffic and with realistic (tool-metadata) predictors. Falsified
if the margin is <10% in that regime on a GPU.

**The trade-off it reveals.** The value of agent awareness at the disk
boundary is set by disk read cost relative to everything else: large on slow
or networked storage, near zero on fast NVMe with a slow GPU or when DRAM
holds the working set.

## State of the art and the gap

| System | Tiers | Agent signal | Disk tier |
|---|---|---|---|
| CacheWise | GPU, CPU | tool name + args, elapsed time | none |
| TokenCake | GPU, CPU | function-call duration, transfer cost, predictive upload | none |
| MORI, EfficientAgent | GPU, CPU | idleness / working set | none |
| Continuum | GPU (+CPU/SSD via LMCache) | tool duration -> GPU TTL | overflow capacity, agent-blind |
| KVTether | GPU, host, remote | message lifecycle, no prediction | reactive |
| Strata, LMCache | GPU, CPU, SSD | none | agent-blind |

No system we found uses agent signals at the CPU->disk boundary. Refs 8 and 9
of the original proposal are not used: both are unreviewed preprints without
code (batch-one synthetic simulation; analytical projections).

## Current results (5 trace samples, after the aging fix)

`kvplace.multiseed`: 5 random samples (seeds 1-5) of 300 TraceLab Claude Code
sessions per setting. Realistic predictor `tool-nfh`: per-tool median of the
last 200 calls, user waits at the 90th percentile, no knowledge of when a
session ends. Every policy gets the same hint traffic; prefetch and demand
share one disk link. p90 resumption TTFT in seconds, mean over samples:

| Setting (GPU / CPU blocks, disk) | LRU | GPU-aware | Ours | Ours vs GPU-aware | Per sample |
|---|---:|---:|---:|---:|---|
| L4 costs, 8k / 48k, 0.93 GB/s | 48.9 | 45.2 | 10.7 | -61% ± 29% | wins 5/5 (-15 to -86%) |
| 12k / 24k, 7 GB/s | 4.27 | 3.76 | 3.35 | -10% ± 7% | wins 5/5 (-5 to -21%) |
| 12k / 96k, 3 GB/s | 2.27 | 2.15 | 1.64 | -21% ± 13% | wins 5/5 (-6 to -42%) |
| 65k / 260k, 3 GB/s | 0.84 | 0.83 | 0.88 | +7% ± 3% | loses 5/5 |

Opportunity gap (12k / 96k, 3 GB/s), reduction vs GPU-aware: oracle 34% ± 3%,
oracle with ~2x error 31% ± 3%, tool median + session-end signal 22% ± 10%,
realistic 21% ± 13%.

**The aging fix.** Sessions with no prediction used to score "last use + 30 s"
forever, so finished sessions looked about to return and were never evicted.
On 2 of 5 samples they filled most of the CPU tier and p90 rose to 36-62 s.
Unpredicted sessions now age like late ones (`ReuseAwareTier._score`,
`ReuseAwareCachePolicy._score`). This also removed the apparent dependence on
a session-end signal, which had only been masking the bug.

**Elapsed-aware prediction (`toolres-nfh`).** Waits are bimodal on a log
scale (instant calls around 0.05 s, then a plateau from 1 s to minutes), so
the hint also carries a per-tool residual table: the median remaining wait of
past calls that already ran e seconds, for e on a log grid. The tier scores a
waiting session as now + remaining(elapsed). Ours, p90 s, same 5 samples:

| Setting | `tool-nfh` | `toolres-nfh` | Ours vs GPU-aware |
|---|---:|---:|---:|
| L4 costs, 8k / 48k | 10.68 | 10.76 | -61% -> -61% |
| 12k / 24k, 7 GB/s | 3.35 | 3.35 | -10% -> -10% |
| 12k / 96k, 3 GB/s | 1.64 | 1.61 | -21% -> -25% |
| 65k / 260k, 3 GB/s | 0.88 | **0.84** | +7% -> **+2%** |

No gain where we already win: the learned residual is roughly e to 2e for
most tools, which the aging rule already assumes. (The -25% comes mostly
from the GPU-aware baseline getting worse on one sample.) It does shrink the
loss when KV fits in DRAM: ours improves on 4 of 5 samples there, none worse.

The single-sample tables below predate the fix and are kept for history.

## Earlier single-sample evidence

All on 300 held-out TraceLab Claude Code sessions (`tl300_sp_s1.jsonl`),
GPU = 12k blocks, prefetch and demand sharing one disk link
(`kvplace.tier_sweep`). "Approx." is the oracle with 2.7x log-normal error.

**Where awareness helps** (p90 resumption TTFT, s, approx. prediction):

| CPU blocks / disk | LRU | GPU/CPU-aware | Demotion-aware | Both | Both vs GPU/CPU-aware |
|---|---:|---:|---:|---:|---:|
| 96k / 3 GB/s | 2.28 | 2.19 | 1.50 | **1.47** | **-33%** |
| 96k / 7 GB/s | 1.44 | 1.37 | 1.22 | **1.18** | **-14%** |
| 24k / 7 GB/s | 3.32 | 2.77 | 2.56 | 2.58 | -7% |

- Most of the agent-aware gain comes from demotion, and it is robust to
  prediction error (approx. is within ~4% of oracle): demotion only needs a
  ranking of sessions, not their timing.
- It fails the bar when the CPU tier is too small to hold the working set.

**With realistic predictors** (every policy, LRU included, sends the same
hint-only requests; `tool` = global tool median, `toolsess` = session-local
per-tool EWMA):

| CPU blocks / disk | LRU | GPU/CPU-aware | Both | Both vs GPU/CPU-aware |
|---|---:|---:|---:|---:|
| 24k / 7 GB/s | 6.26 | 5.09 | **4.09** (`tool`) | **-20%** |
| 96k / 3 GB/s | 2.39 | 2.54 | **1.89** (`toolsess`) | **-26%** |
| 96k / 7 GB/s | 1.36 | 1.34 | 1.22 (`toolsess`) | -9% |

A rough ranking suffices: the global tool median gets most of the gain. Tool
arguments would not be needed (CATraces, the only public trace with tool
calls per session besides TraceLab, redacts them anyway).

**Realistic GPU size** (65k blocks, H100-class; approx. prediction, p90):

| CPU blocks / disk | LRU | GPU/CPU-aware | Both | Both vs GPU/CPU-aware |
|---|---:|---:|---:|---:|
| 130k / 3 GB/s | 1.33 | 1.27 | **1.00** | **-21%** |
| 130k / 7 GB/s | 1.08 | 1.06 | **0.92** | **-14%** |
| 260k / 3 GB/s | 0.90 | 0.88 | 0.84 | -4% |
| 260k / 7 GB/s | 0.88 | 0.85 | 0.84 | -1% |

The ~24 concurrent sessions in the arrival window (~120k blocks of KV) fit in
GPU + 260k CPU blocks (~240 GB DRAM), so little reaches disk and policies
converge. The gain needs a working set larger than DRAM.

**Higher load does not get there** (65k GPU blocks, `tool` predictor, p90):

| Concurrent sessions | CPU blocks / disk | LRU | GPU/CPU-aware | Both | vs GPU/CPU-aware |
|---|---|---:|---:|---:|---:|
| ~37 (0.15/s) | 260k / 3 GB/s | 2.58 | 2.64 | 2.47 | -6% |
| ~37 (0.15/s) | 130k / 7 GB/s | 2.68 | 2.64 | 2.54 | -4% |
| ~45 (0.3/s) | 260k / 3 GB/s | 14.8 | 14.6 | 14.6 | 0% |
| ~45 (0.3/s) | 130k / 7 GB/s | 22.7 | 16.2 | 14.9 | -8% |

Prefill compute saturates before the KV working set overflows DRAM: at ~45
sessions every policy sits at ~14.5 s p90 from queueing. For a 7B GQA model on
a well-provisioned server, the regime where demotion matters is not reached.

**Measured costs (Modal L4, Qwen2.5-7B).** GPU<->CPU 13.5 GB/s, container
disk reads 0.93 GB/s, prefill 3.3k tok/s, decode 57 ms/token. GPU = 8k blocks,
0.01 sessions/s (what one L4 can sustain), p90 with the `tool` predictor:

| CPU blocks / disk | LRU | GPU/CPU-aware | Both | Both vs GPU/CPU-aware |
|---|---:|---:|---:|---:|
| 48k / 0.93 GB/s (measured) | 26.5 | 14.2 | **7.9** | **-45%** |
| 48k / 7 GB/s | 2.88 | 2.91 | 2.87 | -1% |
| 24k / 7 GB/s | 3.16 | 3.31 | 3.35 | +1% |

On the measured (likely network-backed) disk the read link saturates and
demotion choices dominate; with NVMe-speed reads the slow L4 prefill
dominates instead and placement stops mattering. 24k / 0.93 GB/s is
saturated for every policy (p90 70-100 s) and omitted.

**Why not prefetch from disk** (same setup, p90):

| CPU blocks / disk | Demotion only, approx. | + prefetch, oracle | + prefetch, approx. |
|---|---:|---:|---:|
| 24k / 7 GB/s | 2.56 | 2.43 | 3.47 (worse than LRU) |
| 96k / 3 GB/s | 1.50 | 1.05 | 1.50 |

Prefetch only pays with near-exact prediction. With realistic error, wasted
reads delay demand loads on the shared link.

## How we got here (why we pivoted)

1. **Original proposal** (reuse- and cost-aware placement across tiers): the
   mechanism is not new (CacheWise, TokenCake, Continuum; refs 8/9).
2. **First simulator results were inflated.** Hint-only requests warmed the
   GPU for one policy only; with matched traffic the realistic gain over LRU
   fell to 5-14% p90.
3. **Restore timing** (when to bring KV back to GPU): most gaps (p50 0.12 s)
   are shorter than a restore; realistic predictors gained 0-2%
   (`kvplace.restore_sweep`). Dropped.
4. **Trace fidelity fix.** TraceLab's first-call cache hit is a shared system
   prompt; treating it as cold prefill had made the tail look queueing-bound.
   After the fix, KV transfer dominates the tail through p99
   (`kvplace.headroom`).
5. **Other directions checked and dropped as taken:** compaction-aware KV
   (CliffCompaction, TokenPilot, ReCAP, PrefixShield), reasoning-token KV
   (fixed in SGLang/vLLM; Leyline), P/D disaggregation for agents, KV
   overcommit (MORI, TokenCake), agent energy (KAIROS), oracle comparisons
   (InferCept, CacheWise).
6. **Uncontended prefetch was an artifact.** Without link contention, disk
   prefetch looked worth -33% even with noisy predictions; with contention it
   is neutral or harmful. What survives is agent-aware demotion.
7. **One sample hid a policy bug.** Repeating runs on 5 samples exposed the
   unaged unpredicted sessions (see Current results).

## Caveats and next checks

- The regime matters more than the policy. Gains over GPU/CPU-aware
  (5 samples): -61% on the measured slow disk, -10 to -21% on memory-tight
  hosts, +7% (worse) on an H100-size server where KV fits in DRAM.
- Untested regimes where it may matter more: models with larger KV per token
  (no GQA, longer contexts), network-attached cloud disks, more sessions per
  GPU via tensor parallelism.
- Only one GPU type measured (L4). Its container disk is not a local NVMe. An
  H100 with local NVMe is the most relevant missing point.
- Tool-name predictors have large errors (median 3.2x, p90 80x; session-local
  EWMA: rank correlation 0.53 vs 0.43), yet demotion still gains, because it
  needs only a ranking.
- Simulator only: FIFO unchunked prefill, uncontended GPU<->CPU link, no GPU
  working-set pinning. Needs live vLLM replay at one or two map points.
- vLLM implementation: demotion is a `CachePolicy` (exists:
  `vllm_policy.ReuseAwareCachePolicy`); gated prefetch would use the tiering
  manager's promotion path from `on_schedule_end`.

## Reproduce

```bash
export PYTHONPATH=$PWD/research
# current results: traces for seeds 1-5 at 0.05/s and 0.01/s, then
for seed in 1 2 3 4 5; do for r in 0.05 0.01; do
  sfx=$([ $r = 0.01 ] && echo _r0.01)
  .venv/bin/python -m kvplace.tracelab research/traces/syfi_coding_trace.jsonl.gz \
    -o research/traces/tl300_sp_s$seed$sfx.jsonl --provider claude \
    --rebase-rate $r --sessions 300 --max-context 200000 --seed $seed
done; done
(cd research && ../.venv/bin/python -m kvplace.multiseed --out results/multiseed.json)
# earlier single-sample runs
for seed in 0 1; do
  .venv/bin/python -m kvplace.tracelab research/traces/syfi_coding_trace.jsonl.gz \
    -o research/traces/tl300_sp_s$seed.jsonl --provider claude \
    --rebase-rate 0.05 --sessions 300 --max-context 200000 --seed $seed
done
.venv/bin/python -m kvplace.tier_sweep research/traces/tl300_sp_s1.jsonl \
  --policies gpu_aware reuse_evict gpu_cpu_aware reuse
.venv/bin/python -m kvplace.tier_sweep research/traces/tl300_sp_s1.jsonl \
  --predictors tool toolsess
# measured costs (uvx --from modal modal run research/modal_microbench.py --gpu L4)
.venv/bin/python -m kvplace.tracelab research/traces/syfi_coding_trace.jsonl.gz \
  -o research/traces/tl300_sp_s1_r0.01.jsonl --provider claude \
  --rebase-rate 0.01 --sessions 300 --max-context 200000 --seed 1
.venv/bin/python -m kvplace.tier_sweep research/traces/tl300_sp_s1_r0.01.jsonl \
  --cost-json research/costs_l4.json --gpu-blocks 8000 \
  --configs 24000:m 48000:m 24000:7 48000:7 --predictors tool noisy:1.0
.venv/bin/python -m kvplace.headroom breakdown research/traces/tl300_sp_s1.jsonl
.venv/bin/python -m kvplace.restore_sweep --tune research/traces/tl300_sp_s0.jsonl \
  --test research/traces/tl300_sp_s1.jsonl
```
