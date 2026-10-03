# kvplace: reuse-time and transfer-cost aware KV placement

Research code for *Workload and Topology Aware KV Cache Migration on vLLM*.
Everything here is out-of-tree: it plugs into vLLM's `OffloadingConnector`
through `cache_policy_module_path` and request `kv_transfer_params`, and does
not modify `vllm/`. Not intended for upstreaming as-is.

```
research/
  kvplace/
    trace.py          trace schema (JSONL of sessions -> turns -> tool gaps)
    gen_synthetic.py  synthetic coding-agent trace generator
    tracelab.py       converts the public TraceLab trace (real Claude Code sessions)
    stats.py          compares traces (gap tails, autocorrelation, context sizes)
    hints.py          ReuseHint + transports (kv_transfer_params / KvHintsEnvelope)
    predictors.py     none | oracle | noisy:<sigma> | ewma[:<alpha>] | tool
    sim.py            offline 3-tier simulator (week-2 go/no-go)
    vllm_policy.py    ReuseAwareCachePolicy for vLLM's CPU tier
    replay.py         replays a trace against a live vLLM server
    microbench.py     GPU<->CPU, FS, and prefill/decode cost measurement
  scripts/run_baselines.sh   week-1 baseline matrix on the GPU box
  tests/              simulator + policy tests (CPU-only, run anywhere)
```

## Setup

```bash
# from the vllm repo root
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto   # GPU box
# VLLM_TARGET_DEVICE=empty uv pip install -e .                    # laptop: sim + tests only
uv pip install pytest aiohttp requests
export PYTHONPATH=$PWD/research
.venv/bin/python -m pytest research/tests -q
```

## Week 1: baselines and costs (GPU box)

1. Pin the commit: record `git rev-parse HEAD` (the launcher writes it into
   every results directory). Don't rebase during the project.
2. Generate the trace:
   ```bash
   .venv/bin/python -m kvplace.gen_synthetic -o research/traces/syn300.jsonl \
       --sessions 300 --rate 0.3
   ```
3. Measure costs into one JSON (`CostModel` fields):
   ```bash
   .venv/bin/python -m kvplace.microbench gpu --out research/costs.json
   .venv/bin/python -m kvplace.microbench fs --dir /mnt/nvme/kvplace_bench --out research/costs.json
   # with a plain `vllm serve $MODEL --block-size 16` running:
   .venv/bin/python -m kvplace.microbench server --out research/costs.json
   ```
4. Run the baseline matrix (`gpu_only cpu_lru cpu_arc tier_lru tier_arc`):
   ```bash
   TRACE=research/traces/syn300.jsonl FS_DIR=/mnt/nvme/kvplace_fs \
       research/scripts/run_baselines.sh
   ```
   Run it twice. **Exit criterion:** resumption-TTFT p50/p90 within ~5%
   between runs. If not, find the noise source (other GPU tenants, page cache,
   CPU frequency scaling) before going further.
5. Calibrate the simulator: run `kvplace.sim` with `--cost-json
   research/costs.json` and the same GPU/CPU sizes, and compare its `lru`
   resumption TTFT and tier-hit fractions with `tier_lru`. Large gaps mean a
   simulator assumption is wrong (see Simulator limitations).

Sizing: `GPU_BLOCKS` sets prefix-cache pressure directly
(`--num-gpu-blocks-override`). The CPU tier in blocks is
`CPU_BYTES / (block_size * kv_bytes_per_token)`, which is ~0.92 MB per 16-token
block for Qwen2.5-7B. The default 10 GB is about 10.9k blocks, close to the
`--cpu-blocks 12000` simulator setting below.

## Week 2: design + go/no-go

### Hint design (decided)

- **Payload** (`kvplace.hints.ReuseHint`): `session_id`, `turn`,
  `expected_reuse_s` (seconds from *this request finishing* to the session's
  next request), and `final`.
- **Transport:** `kv_transfer_params["kvplace"]` over HTTP. vLLM already has a
  first-class `KvHintsEnvelope` (`vllm/v1/kv_hints/protocol.py`), but it is
  only reachable through the Python engine API, so `parse_hint` accepts both
  and prefers the envelope. Both arrive in the policy via `ReqContext`.
- **Timing gap, resolved with hint-only requests:** the agent learns which
  tool it's calling only *after* the response, but hints are attached at
  submission. With a post-response predictor (`tool`), the turn's request
  carries no hint. Right after the response, `replay.py` sends a **hint-only
  request**: the same prompt plus padding up to one new offload block,
  `max_tokens=1`, with the hint. It runs concurrently with the tool and never
  delays the next turn.
  - **Why the padding:** the tiering manager's `on_new_request` doesn't create
    the CPU tier's per-request state; that only happens on a CPU store or
    load. A hint-only request whose prefix is all GPU hits would therefore
    never reach `CachePolicy.on_request_finished`, and the hint would be
    silently dropped. `test_storeless_request_hint_needs_on_new_request`
    documents this. Storing one fresh block fixes it.
  - **Costs:** one extra request per turn, a ~16-token prefill, and one junk
    block per turn in CPU/NVMe. If GPU blocks were evicted, the prefix may also
    be loaded early, which is effectively a free prefetch.
  - **Checking it on a server:** run with `KVPLACE_LOG_HINTS=1` and confirm
    one "kvplace hint" log line per hint-only request.
  - **The `tool` predictor** gives the median recent duration of the chosen
    tool, learned online across sessions. The `final` flag is exact, because a
    response with no tool call ends the session. Selective writes still only
    see submission-time hints, since stores happen before the response.

### Policy (first version; week 3 hardens it)

| Decision | Rule | Where |
|---|---|---|
| CPU eviction | Evict the session with the farthest predicted next use (final > overdue > hinted > unhinted+30s), prefix tail first; chunks shared across sessions last | `vllm_policy.py`, `sim.ReuseAwareTier` |
| Secondary write | Final turn: skip. Reuse < 5 s: defer, write back only if evicted. Otherwise write-through | sim only (week 3: tiering manager) |
| Promotion | Secondary->CPU at `deadline - (transfer time + 0.5 s)` | sim only (week 3: `on_schedule_end`) |

### Go/no-go on the synthetic trace (superseded by the real-trace results below)

300 sessions, 3.3k turns, GPU = 3,000 blocks, unbounded FS tier.
Resumption TTFT p90 in seconds (change vs LRU):

| CPU blocks | LRU | reuse+oracle | reuse+noisy σ=1 | reuse+ewma |
|---|---|---|---|---|
| 6,000 | 0.516 | 0.432 (−16%) | 0.477 (−8%) | 0.497 (−4%) |
| 12,000 | 0.457 | 0.278 (−39%) | 0.374 (−18%) | 0.462 (+1%) |
| 24,000 | 0.342 | 0.269 (−21%) | 0.293 (−14%) | 0.403 (+18%) |
| 48,000 | 0.287 | 0.269 (−6%) | 0.269 (−6%) | 0.298 (+4%) |

What this says:
- **Go.** There's real headroom when the CPU tier is about 2–8× the GPU prefix
  cache. When the CPU tier holds everything, nothing is gained (48k row).
- **The predictor matters most.** Accurate hints give most of the gain, and
  ~2.7× multiplicative error keeps about half of it. (The EWMA result here is
  an artifact of the synthetic generator; see the real trace section below.)
- **Secondary write traffic drops 36–77%** from skipping final turns and
  deferring short-reuse writes.
- **Prefetch thrashes under tight CPU:** at 6k blocks, 54% of prefetched
  blocks are evicted before use. Week 3 needs admission control: don't promote
  if that evicts blocks with an earlier deadline.

### Real trace: TraceLab (Claude Code subset)

[TraceLab](https://github.com/uw-syfi/TraceLab) (CC BY 4.0) has 8k real
Claude Code / Codex sessions with per-call token counts, event timestamps and
per-tool wall latency. Convert it with `kvplace.tracelab`, which documents the
field mapping, and compare traces with `kvplace.stats`.

```bash
curl -fLo research/traces/syfi_coding_trace.jsonl.gz \
  https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz
# full scale (simulator only); real contexts don't fit a 32k model
.venv/bin/python -m kvplace.tracelab research/traces/syfi_coding_trace.jsonl.gz \
  -o research/traces/tl300_full_r05.jsonl --provider claude \
  --rebase-rate 0.05 --sessions 300 --max-context 200000
# scaled to fit Qwen2.5-7B (32k) for replay on the GPU box
.venv/bin/python -m kvplace.tracelab research/traces/syfi_coding_trace.jsonl.gz \
  -o research/traces/tl300.jsonl --provider claude --token-scale 0.25 \
  --max-context 30000 --rebase-rate 0.3 --sessions 300
```

**Why Claude only:** for Claude rows, the inter-round gaps match the recorded
tool latencies (p50/p90/p99 0.12/11.8/258 s vs 0.09/11.2/260 s). Codex gaps
(p50 2.0 s) don't match its tool latencies (p50 0.6 s), so its timestamp
semantics are unclear. `prefix_tokens` is the provider's *cache hit*, not the
logical prefix, so it isn't used.

**How the synthetic generator compares:**

| | Synthetic | TraceLab Claude |
|---|---|---|
| gap p50 / p90 / p99 | 0.6 / 46 / 241 s | 0.17 / 120 / 1,998 s |
| gaps > 60 s | 8% | 13% |
| lag-1 corr. of log(gap) | 0.01 | **0.31** |
| final context p50 / p90 | 13k / 26k | 84k / 366k |
| "tools" | read_file, edit, grep… | Bash, Read, **human (12%)**, Edit |

The synthetic generator's independent gaps made EWMA look useless. Real gaps
are autocorrelated, and human think time (minutes to hours) is a major class.
Use TraceLab-derived traces for all results from now on.

**Go/no-go on the real trace** (full token scale, 300 sessions at 0.05/s,
GPU = 12k blocks, default costs). Resumption TTFT in seconds, p50 / p90:

| CPU blocks | LRU | oracle | noisy σ=1 | tool (hint-only) | ewma |
|---|---|---|---|---|---|
| 24,000 | 1.89 / 4.32 | 0.44 / 2.45 | 0.51 / 3.33 | 0.52 / 3.41 | 0.82 / 3.82 |
| 48,000 | 1.06 / 4.10 | 0.43 / 2.04 | 0.48 / 3.20 | 0.49 / 3.21 | 0.57 / 3.65 |
| 96,000 | 0.49 / 3.70 | 0.44 / 2.03 | 0.46 / 2.93 | 0.46 / 2.93 | 0.50 / 3.42 |

- **Still a go, and a stronger one than synthetic.** At real context sizes,
  LRU serves ~50–70% of reused blocks from NVMe at 24k–48k CPU blocks. With
  the hint-only `tool` predictor, p50 drops 54–73% there. At 96k, LRU
  already keeps most blocks in CPU, so the p50 gain vanishes (−7%), but p90
  still drops ~21% at every size. p99 (~12–14 s) barely
  moves because it's dominated by prefill of large new inputs, not placement.
- **Tool-aware hints get most of the oracle's p50 gain but only half its p90
  gain.** The long tail of the gap distribution (human waits) is where
  prediction is weakest.
- **Caveat:** at 4× token scale (`tl300.jsonl`, fits 32k models), gains shrink
  to −8% p90 for `tool`. Transfer volume scales with context, so a short-context
  replay will understate the effect. Consider a long-context model (e.g.
  Llama-3.1-8B, 128k) for the GPU runs.
- At 0.3 sessions/s and full scale, the simulated GPU is overloaded (p90 75 s).
  Keep load sustainable, or the queueing will hide the placement effects.

### Simulator limitations

- No GPU working-set pinning. The GPU tier is treated as pure prefix-cache
  capacity.
- Cache updates are applied at request arrival, not at prefill completion.
- Prefill is FIFO on one GPU, and decode doesn't contend for it.
- Transfers are uncontended. Prefetch bandwidth is counted but doesn't delay
  demand loads.
- The model's real outputs are replaced by synthetic tokens in the next turn
  (as in `replay.py`), so decode KV is never reused.
- No ARC in the simulator. Compare against ARC on the real system.
- Default `CostModel` numbers are placeholders until `microbench` runs.

## Week 1–2 checklist

- [x] Trace schema + synthetic generator
- [x] Offline simulator with LRU / reuse-aware CPU tier, prefetch, selective write
- [x] Hint format + transport decided; predictors (oracle/noisy/ewma)
- [x] `ReuseAwareCachePolicy` loads in vLLM's real `CPUOffloadingManager`
      (tests fail under LRU, pass under the policy)
- [x] Replay client (validated against a mock server), microbenchmarks, launcher
- [ ] **GPU box:** microbench → `costs.json`; baselines ×2 for reproducibility
- [ ] Rerun the go/no-go table with measured costs; calibrate sim vs `tier_lru`
- [x] Hint timing: hint-only requests + `tool` predictor (sim + replay + tests)
- [x] Public real trace (TraceLab) converted, compared, go/no-go rerun
- [ ] **GPU box:** confirm hint-only requests reach the policy (`KVPLACE_LOG_HINTS=1`)
- [ ] Decide model/context: 32k with token scaling vs a 128k model
