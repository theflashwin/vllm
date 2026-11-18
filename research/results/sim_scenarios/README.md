# Simulator scenarios (Nov 16)

Traces in `traces/tl70/`, built from the public TraceLab v0.0.2 file
(`traces/syfi_coding_trace.jsonl.gz`, SHA-256 in the main README). Same 70
sessions and 2,530 turns in each; only the named setting differs.

```bash
TL="python -m kvplace.tracelab traces/syfi_coding_trace.jsonl.gz --provider claude \
  --rebase-rate 0.02 --sessions 70 --seed 1"
$TL --token-scale 0.15 --max-context 32000 --max-gap-s 600  -o traces/tl70/tl70_base.jsonl
$TL --token-scale 0.3  --max-context 64000 --max-gap-s 600  -o traces/tl70/tl70_longctx.jsonl
$TL --token-scale 0.15 --max-context 32000 --max-gap-s 3600 -o traces/tl70/tl70_longwait.jsonl
```

Costs: `costs_scan_a100.json` with Llama 3.1 8B KV (131072 bytes per token) and
the disk at 2.1 or 0.25 GB/s for reads and writes (`cost_llama_disk*.json`).

```bash
python -m kvplace.sim <trace> --policies lru reuse_evict --predictors oracle tool-nfh2 \
  --gpu-blocks <3800, or 7600 for longctx> --cpu-blocks <2000|6000|10000> \
  --time-scale <1.0, or 2.0 for live70_ts0.35> --gpu-admission --contend-sec \
  --write-model <separate|shared> --cost-json cost_llama_disk<2.1|0.25>.json
```

Output files are `<trace>_<disk GB/s>_<write model>_<cpu blocks>.tsv`;
`live70` is `traces/live/live70_ts0.35.jsonl`. One run per cell. At 2000 blocks
with a 0.25 GB/s disk the simulator overloads (p50 of minutes), which the live
runs at 4.19 GB did not (p50 0.11 s), so those cells overstate disk contention.
