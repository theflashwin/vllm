# Live vs simulator: paired100 on A100

Simulator prediction for the live run in `modal_live.py` (`tier_lru` vs
`tier_reuse`), to compare against the live results once they are in.

## Setup

Same inputs as the live run:

| Input | Value |
|---|---|
| Trace | `paired100.jsonl` (sha256 `3036208cb109be74…`) |
| GPU / CPU | 3000 GPU blocks, 10 GB CPU tier (10898 blocks) |
| Disk | one shared link for reads and writes (`--contend-sec --write-model shared`) |
| Costs | `costs_a100.json` from this PR |
| Predictor | `tool-nfh2`, time scale 0.2 |
| Policies | GPU LRU in both; CPU tier `lru` vs `reuse_evict` |

```bash
cd research
PYTHONPATH=. python -m kvplace.tracelab traces/syfi_coding_trace.jsonl.gz \
  -o traces/live/paired100.jsonl --provider claude --token-scale 0.25 \
  --max-context 30000 --max-gap-s 300 --sessions 100 --rebase-rate 0.05 --seed 1
PYTHONPATH=. python -m kvplace.sim traces/live/paired100.jsonl \
  --gpu-blocks 3000 --cpu-blocks 10898 --policies lru reuse_evict \
  --predictors tool-nfh2 --cost-json costs_a100.json --contend-sec \
  --write-model shared --time-scale 0.2 --out results/live/sim_paired100_a100.json
```

## Prediction

Resume TTFT over 1627 resumed turns:

| CPU policy | p50 | p90 | p99 | Resumes from CPU | Resumes from disk |
|---|---:|---:|---:|---:|---:|
| LRU | 0.21 s | 23.3 s | 26.8 s | 6.0% | 4.2% |
| Ours | 0.14 s | 21.2 s | 27.5 s | 8.1% | 3.0% |
| Ours vs LRU | -32% | -9% | +2% | | |

- Ours moves resumes from disk to CPU, as intended.
- The tail is not set by eviction. In the slowest 15% of resumes, ~22 s of
  the ~24 s TTFT is `warmup_wait_s`: the turn waits for the previous
  hint-only request's prefix load, queued on the disk link behind write
  traffic. Queueing, load and prefill add only ~2 s.

## What to check live

- p90 within ~10% of the simulator for both policies, and the same sign of
  the policy gap.
- Whether live shows the same ~20 s tail. If not, the simulator overstates
  how much hint-only loads and writes block the next turn.
- Resumes from CPU vs disk, from `metrics_delta.json` (bytes read per tier).
- `server.log`: hints logged by the policy, and no O_DIRECT fallback warning.
