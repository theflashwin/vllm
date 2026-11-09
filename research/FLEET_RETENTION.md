# Fleet effect of client cache retention

Status: simulator only, Qwen2.5-7B on A100 costs, one GPU. No live check yet.
Code: `kvplace/retention_sweep.py`, `Retention` in `kvplace/sim.py`. Results:
`results/fleet/` (gitignored).

## Question

When clients buy long cache tiers or send keepalive pings, what happens to provider
memory, capacity and other users' latency? [Keeping the Cache Warm
Pays](https://arxiv.org/html/2607.19214) argues keepalives degrade the tier for
everyone, but simulates nothing at fleet level. [When Cache Pressure Becomes a
Trap](https://arxiv.org/html/2611.02543) shows the shared-cache feedback after surges,
without client retention choices, provider tiers or dollars.

## Model

- TTL: a session's blocks leave every tier `ttl_s` after its last read. LRU capacity
  eviction still applies. Sessions not adopting a plan use 5 minutes.
- Keepalive: a real request with the full prefix every 270 s, one outstanding, up to
  1 h after the last response. It pins its context on GPU and promotes it there.
- `notail`: no pings after the session's last turn. `touch`: the provider refreshes
  the TTL and LRU recency without running a request.

## Capacity at p90 TTFT <= 5 s, change from everyone on 5 minutes

TraceLab mean of 5 samples, per-sample range in brackets; CATraces is one trace.

| Everyone on | TraceLab, 128 GB + SSD | TraceLab, 128 GB, no SSD | CATraces, + SSD | CATraces, no SSD |
|---|---:|---:|---:|---:|
| 30 min TTL (2 TraceLab samples) | +14% [+10,+19] | +12% [+9,+14] | +13%* | 0% |
| 1 h TTL | +18% [+11,+28] | +12% [+8,+20] | +13%* | 0% |
| Keepalive 1 h, no dead-session pings | +17% [+11,+27] | +11% [0,+21] | +13%* | -20% |
| Keepalive 1 h, refresh only | +17% [+11,+27] | 0% [-8,+8] | +13%* | -20% |
| Keepalive 1 h | +8% [+4,+11] | -27% [-32,-21] | -28% | -100% |

*Reached the top of the rate grid (0.015), so the true gain may be larger.
Base capacity: TraceLab 0.0049 sessions/s, CATraces 0.0133 with SSD and 0.0125 without.
At 64 GB and no SSD, TTLs change nothing and full keepalives fall below the lowest rate
tested (2 TraceLab samples and CATraces; coarse grid).

## Other users' latency, half the clients adopting

Median of 5 TraceLab samples, p90 TTFT of non-adopters at 0.005 sessions/s, against 6.0 s
when nobody adopts:

| Half the clients on | 128 GB + SSD | 128 GB, no SSD |
|---|---:|---:|
| 1 h TTL | 3.9 s | 3.9 s |
| Keepalive 1 h | 4.3 s | 10.6 s (at least one sample collapses; all-user p90 up to 1782 s) |

Near the limit, full keepalives can push the system into a collapse: misses rise, requests
queue and memory fills. CATraces at 0.01 without SSD: others 489 s vs 1.3 s.

## Reading

- Holding longer did not hurt others here. Fewer recomputes appear to leave more GPU for
  everyone, until memory runs out, and then long TTLs simply stop helping.
- The harm comes from how retention is obtained. At 128 GB without SSD, moving from full
  keepalive to refresh only recovers 27 points, and dropping the recency refresh as
  well (a plain 1 h TTL) another 12. This matches the "manufactured recency" argument, but
  request load is the bigger part.
- 30 minutes gets most of the gain at about half the extra DRAM of 1 h.
- Dead sessions are about 70% of ping tokens on TraceLab (0.51 vs 0.16 Gtok).

## Provider cost, TraceLab at 0.004 sessions/s

From 300 sessions, about 21 h. Assumed prices: A100 $2/h ($42 for the run), DRAM $0.0088
and NVMe $0.000082 per GB-hour, if retained memory is charged only while occupied.

| Tier | DRAM | NVMe |
|---|---:|---:|
| 5 min | $7.89 | $0.07 |
| 30 min | $15.13 (2 samples) | $0.15 (2 samples) |
| 1 h | $22.06 | $0.22 |

An 18% capacity gain is worth about $6.40 of GPU time (15% fewer GPUs). That is below the
$14 of extra DRAM for 1 h, and far above the $0.15 of extra NVMe. These compare hardware
costs only: no margin, prices or other model shapes.

## Limits

- Only the 7B model; prefill cost dominates these results, and misses block the GPU.
- The simulator reads latency about 2-2.5x low against live runs, so the capacity
  points are not absolute. Comparisons between arms are the claim.
- Ping decode contention is not modeled, which understates ping cost slightly.
- A trace's last turn may not be a truly dead session, because of compaction splits and
  collection limits, so the tail share may be overstated.
- One GPU. Provider tiers are assumed to expire exactly at the TTL.
