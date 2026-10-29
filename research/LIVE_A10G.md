# Live vs simulator on A10G (first pass)

Status: one live run (LRU) completed; the simulator does not match it yet.
The mismatch points at two things the simulator does not model.

## Setup

| Input | Value |
|---|---|
| Trace | `live70`: first 70 sessions of a TraceLab sample, token scale 0.15, max context 32k, waits capped at 600 s |
| GPU / CPU | 3800 GPU blocks (pinned), 3.67 GB CPU tier (4000 blocks), fs disk tier |
| Predictor, time scale | `tool-nfh2`, 0.2 |
| Costs | `costs_a10g.json` (measured): prefill 4.0k tok/s, decode 32.7 ms/token, GPU<->CPU 25.3 GB/s, disk 1.49 GB/s |

```bash
KVPLACE_GPU=A10G modal run research/modal_live.py --trace research/traces/live/live70.jsonl \
  --configs tier_lru --gpu-blocks 3800 --cpu-gb 3.6701 --run live70_a10g_r2
PYTHONPATH=. python -m kvplace.sim traces/live/live70.jsonl --gpu-blocks 3800 \
  --cpu-blocks 4000 --policies lru reuse_evict --predictors tool-nfh2 \
  --cost-json costs_a10g.json --contend-sec --write-model shared --time-scale 0.2
```

## Results

Resume TTFT, CPU tier LRU:

| | p50 | p90 | p99 | Wall time |
|---|---:|---:|---:|---:|
| Simulator | 0.06 s | 0.87 s | 2.3 s | ~30 min |
| Live | 14.7 s | 44.9 s | 60.9 s | 55 min |

Simulator only, ours vs LRU (p90): -10% at token scale 0.15, -26% at 0.25,
-20% at 0.35. The live runs with our policy and at token scale 0.25 were lost
when the launching session ended (no `--detach`).

## Why live is ~50x slower

1. **GPU memory admission.** vLLM logged KV cache usage at 90-100% with 2-4
   requests running and 20-57 waiting. Running requests hold their whole
   context in the same 3800 blocks, so only ~5 fit at once. The simulator
   treats GPU blocks as a cache only and prefills one request at a time, so it
   never sees this queue.
2. **Disk thrash.** The fs tier read 3.54 TB (~1.08 GB/s, near the disk's
   1.49 GB/s) and wrote 0.44 TB, while only 0.67 TB moved CPU->GPU, about 60x
   the trace's ~55 GB of unique KV. The 3.67 GB CPU tier is too small to hold
   blocks staged from disk until they reach the GPU, so they are evicted and
   read again. The simulator loads each needed block once.

Neither is about the eviction policy, so this setting cannot test it yet.

## Other findings

- **Replay connection bug (fixed here).** vLLM closes idle keep-alive
  connections after 5 s; replay sessions sleep longer than that between turns,
  and aiohttp reused closed connections (`ServerDisconnectedError` ~10 s into
  every run). `replay.py` now uses `force_close=True`.
- **Hint-only requests cost latency.** In the simulator at token scale 0.25,
  ~0.14-0.19 s of a median turn is waiting for the previous hint-only request
  to finish loading its prefix. Server-side prediction would remove it.

## Next

1. Simulator: running requests hold their blocks on GPU, with an admission
   limit like vLLM's; model staged disk reads competing for CPU space.
2. Live: rerun (detached) with the A10G's natural KV budget and a CPU tier that
   can stage loads, so disk loads rather than admission dominate; then compare
   both policies against the fixed simulator.
