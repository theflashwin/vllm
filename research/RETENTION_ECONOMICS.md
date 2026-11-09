# Prompt-cache retention economics for agents

Status: direction chosen (Nov 2026); client-side and provider-side first
passes done on fixed traces; simulator fixes done; fleet simulation next.

## Question

Agent providers sell KV-cache retention as tiers: Anthropic 5-minute (write
1.25x base input) and 1-hour (2x) caches with reads at 0.1x that refresh the
TTL; OpenAI 30-minute (1.25x) and 12-hour (2.5x) on newer models; Gemini
charges storage per token-hour. Clients can also send keepalive requests.

1. **Client side:** how much do agents save by choosing a tier and keepalives
   per call, using predicted waits?
2. **Provider side:** what does holding KV cost in GPU memory, CPU memory and
   SSD compared with recomputing it, and where do the tiers fall?
3. **Fleet (systems):** what do keepalives and long TTLs do to a provider's
   memory pressure and to other users' latency?

Closest work: [Keeping the Cache Warm Pays](https://arxiv.org/abs/2607.19214)
(fixed-timer keepalive economics on synthetic probes; fleet effect argued,
not simulated), [Can I Buy Your KV Cache?](https://arxiv.org/abs/2606.13361)
(reuse 9-50x cheaper than prefill), [Don't Break the Cache](https://arxiv.org/abs/2601.06007)
(breakpoint placement). See `FUTURE_DIRECTIONS.md` for the full review.

## 1. Client side (first pass)

Input-token cost vs "always use the short tier". Per turn: hit = read the
cached prefix + write new tokens at the tier price; miss = re-write the whole
prompt. Keepalive = a read every TTL - 30 s while idle, if cheaper than a
re-write. Per-tool tier choice is learned on the other samples (TraceLab:
leave-one-sample-out over 5 samples; CATraces: trained on TraceLab).

| Policy | Anthropic 5m/1h, TraceLab | Anthropic, CATraces | OpenAI 30m/12h, TraceLab | OpenAI, CATraces |
|---|---:|---:|---:|---:|
| No caching | +516% | +684% | +638% | +752% |
| Always long tier | -6.7% | -0.4% | +14.2% | +10.9% |
| Short tier + keepalive | -14.1% | -7.1% | -4.7% | -1.7% |
| Per-tool tier | -11.5% | -1.7% | -3.3% | +0.9% |
| Per-tool tier + keepalive | -18.0% | -7.7% | -6.3% | -2.0% |
| Oracle (best per gap) | **-19.0%** | **-8.7%** | **-6.7%** | **-2.5%** |

Rerun on traces from the fixed converter. TraceLab range across samples: oracle
-16% to -26% (Anthropic).

- **Caching itself is worth 5-8x; tier choice is worth at most ~19%.** 98-99%
  of cached prefix tokens return within 5 minutes.
- **Keepalive captures most of the remaining saving;** prediction adds about
  4 points on TraceLab and under 1 on CATraces. A client-side "choose the tier
  by predicted wait" contribution is weak on its own.
- **Longer short tiers leave less to optimize:** with OpenAI's 30-minute short
  tier the ceiling is under 7%.

Caveats: input-token cost only; prices as of Nov 2026; TTL assumed to be
chosen after the response (if it must be chosen at write time, before the
tool call is known, savings shrink further).

## 2. Provider side (first pass)

Break-even holding time: how long keeping one token's KV in a tier stays
cheaper than recomputing it. Assumptions: A100 at $2/hour with 40 GB HBM,
measured Qwen2.5-7B prefill 11.3k tok/s, the same $/FLOP for larger models;
DRAM $0.0088 and NVMe $0.000082 per GB-hour (Tutti, arXiv 2605.03375).

| Model shape | HBM | DRAM | SSD |
|---|---:|---:|---:|
| 7B dense GQA (Qwen2.5-7B, measured) | 62 s | 5.9 min | 10.5 h |
| 70B dense GQA (Llama-3-70B shape) | 108 s | 10.3 min | 18.3 h |
| MoE + MLA (DeepSeek-V3 shape) | 4.5 min | 25 min | 45 h |

- **Holding in GPU memory pays only for 1-5 minutes,** so a 5-minute tier is
  already beyond HBM break-even for dense models.
- **The 5-minute tier is about DRAM break-even; 1-hour and longer tiers only
  pay on SSD** (or for models with very compact KV). Tiers roughly mirror the
  storage hierarchy.
- **A keepalive every ~4.5 minutes converts a client's would-be miss into an
  indefinite DRAM-class residency** at 0.1x per ping, about 1.3x the base
  input price per hour of idle time.

Extending retention from 5 minutes to 1 hour (fixed traces):

| Model shape | Extra holding, in DRAM | Extra holding, on SSD | Extra recompute avoided |
|---|---:|---:|---:|
| 7B dense | 19-39% of total recompute cost | 0.2-0.4% | 1.0-2.8% |
| 70B dense | 11-22% | 0.1-0.2% | 1.0-2.8% |
| MoE + MLA | 4-9% | <0.1% | 1.0-2.8% |

Ranges are CATraces to TraceLab. A 1-hour tier held in DRAM costs far more than
it saves; on SSD it about breaks even or better. This excludes reload bandwidth,
contention and any latency effect, so it is an upper bound on the placement win.

Caveats: frontier model sizes and serving costs are not public; prices
include margin; FP8 or compressed KV shifts the numbers; reload bandwidth and
contention are ignored.

## 3. Fleet simulation (next)

Extend the simulator with provider-side retention obligations: each client
policy pins KV until its TTL expires (refreshed by reads and pings), placed in
HBM, DRAM or SSD. Measure: memory and storage footprint, hit rates, and the
TTFT of other traffic, as the share of clients using long tiers or keepalives
grows. Then test placement policies (which tier holds which TTL) and prices
that recover holding cost.

Simulator fixes are done (one GPU memory pool with a FIFO admission queue,
shared-prefix cap, compaction-only session splits, warm-up timing, paired
routing). Remaining gap against live runs: latency about 2-2.5x too low, and no
promotion thrash (see "Simulator status").

## Simulator status

With admission, on A10G (LRU unless noted):

| Run | Simulator p50 / p90 / p99 | Live p50 / p90 / p99 |
|---|---:|---:|
| `live70`, 3.7 GB CPU | 8.7 / 22.1 / 30.3 s | 17.8 / 43.0 / 58.8 s |
| `live70`, 14.7 GB CPU | 4.8 / 19.6 / 28.2 s | 11.0 / 27.5 / 39.2 s |
| `live35`, stable | 0.04 / 0.32 / 3.6 s | 0.21 / 0.79 / 9.3 s |
| `live35`, stable, ours | 0.04 / 0.35 / 4.1 s | 0.21 / 1.23 / 13.8 s |

Disk reads: simulator 630 GB vs live 4,038 GB at 3.7 GB CPU (85% of live
promotions were wasted, not modeled); 3 GB vs 35 GB at 14.7 GB CPU. The direction
of each comparison is right (larger CPU tier faster, ours slightly worse than
LRU when stable), but ours' live loss (+56% p90) is larger than simulated (+9%).
Not modeled: promotion thrash, batched decode slowdown, per-request API and
scheduling overhead (`sched_overhead_s` is 0 in the measured costs).

## Plan

1. A few dollars of live API probes: does a post-response 1-hour refresh bill
   as a read or a re-write? (Decides whether tier choice can use the tool call.)
2. Calibrate the simulator's remaining gap against the live runs.
3. Fleet simulation (part 3), the main systems contribution.
4. Placement and pricing proposal; sensitivity to model shape and KV format.
