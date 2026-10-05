# Project direction: agent-aware demotion to the disk tier

Status as of 2026-10-05. Simulator results only; nothing below is validated on
a GPU yet. Costs are placeholders until `microbench` runs.

## Thesis

Agent-aware KV systems (CacheWise, TokenCake, MORI, EfficientAgent) decide
what stays on the GPU versus CPU. On hardware with an NVMe tier, the decision
that matters more is which sessions the CPU tier demotes to disk. Putting
agent awareness there, and *not* using it to prefetch from disk over a
contended link, reduces tail resumption latency beyond GPU/CPU-only
agent-aware policies.

**Hypothesis.** With predicted reuse times, agent-aware CPU->NVMe demotion
reduces p90 resumption TTFT by >=10% over a GPU/CPU-only agent-aware policy,
with no extra disk traffic. Falsified if the margin is <10% on a GPU.

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

## Evidence

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

## Caveats and next checks

- The simulated GPU is small (12k blocks, ~190k tokens). A realistic H100 with
  a 7B model holds ~65k blocks, which gives GPU-side awareness more room and
  may shrink our margin. **Next check.**
- "Approx." is unbiased noise with an exact session-end flag. A real predictor
  is needed: tool-name medians are much worse (median error 3.4x, p90 80x),
  mostly from Bash. An argument-aware predictor needs a trace with tool
  arguments (CATraces, `cachewise-project/cachewise-coding-traces`).
- The gain is moderate (0.2-0.8 s at p90) and mostly in the tail.
- Simulator only: placeholder costs, FIFO unchunked prefill, uncontended
  GPU<->CPU link. Needs Modal microbenchmarks and live vLLM runs.
- vLLM implementation: demotion is a `CachePolicy` (exists:
  `vllm_policy.ReuseAwareCachePolicy`); gated prefetch would use the tiering
  manager's promotion path from `on_schedule_end`.

## Reproduce

```bash
export PYTHONPATH=$PWD/research
for seed in 0 1; do
  .venv/bin/python -m kvplace.tracelab research/traces/syfi_coding_trace.jsonl.gz \
    -o research/traces/tl300_sp_s$seed.jsonl --provider claude \
    --rebase-rate 0.05 --sessions 300 --max-context 200000 --seed $seed
done
.venv/bin/python -m kvplace.tier_sweep research/traces/tl300_sp_s1.jsonl \
  --policies gpu_aware reuse_evict gpu_cpu_aware reuse
.venv/bin/python -m kvplace.headroom breakdown research/traces/tl300_sp_s1.jsonl
.venv/bin/python -m kvplace.restore_sweep --tune research/traces/tl300_sp_s0.jsonl \
  --test research/traces/tl300_sp_s1.jsonl
```
