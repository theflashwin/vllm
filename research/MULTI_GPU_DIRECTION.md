# Direction: agent-aware KV management across GPUs sharing a KV pool

Status: proposal, based on simulator results with GPU admission modeled
(`ADMISSION_FINDINGS.md`). Needs a novelty check and confirmation that it
holds with realistic storage speeds.

## Why move here

On a single server at a load it can sustain, the CPU tier barely matters for
agent traffic. The GPU prefix cache already holds the sessions that return
soon, and sessions that return late have left CPU too. Removing the CPU tier
entirely gives identical latency on both public traces (TraceLab, CATraces).
So no CPU-tier eviction policy can help there.

Production serving is rarely one GPU. Systems like LMCache, Mooncake and
llm-d run several GPU servers over a shared CPU/disk KV pool, behind a
router. Then a session's next turn can land on a GPU that never saw it, and
the shared pool, not the GPU cache, decides whether it reloads quickly.
That is the only setting where we saw the CPU tier matter.

## Evidence so far

Simulator, 4 A100-80-class GPUs at 4x the single-GPU sustainable load
(0.08 sessions/s), shared CPU pool and one shared 0.96 GB/s disk link,
`tool-nfh2` predictions, 5 trace samples:

| Shared pool | Affinity | LRU p90 | Ours vs LRU p90 | Samples better |
|---|---|---:|---:|---:|
| 29 GB | 50% | 61 s | -46% +/- 34% | 4/5 |
| 59 GB | 25% | 46 s | -50% +/- 42% | 4/5 |
| 59 GB | 50% | 11 s | -20% +/- 41% | 3/5 |
| 29 GB | 75% | 2.1 s | -5% +/- 31% | 2/5 |
| 59 GB | 75% | 0.8 s | +2% +/- 14% | 2/5 |

"Affinity" is the chance a session's next turn goes back to the same GPU.

- **Affinity is the dominant knob.** At 100% affinity this is the single-GPU
  case again and there is no gain.
- **The gain comes from disk contention.** In the winning cells 6-12% of
  resumes load from disk and the four GPUs saturate the shared disk link. Our
  policy reads less from disk, which also shortens the queue for everyone.
- **It is fragile.** Variance is large, the best cells are near the edge of
  stability, and a small pool with low affinity is worse with our policy.

## The research question

> When load balancing breaks session affinity, how much of the lost cache
> benefit can agent-aware routing and pool management recover, and which
> matters more?

Two levers, which should be designed together:

1. **Routing.** Send a returning session back to the GPU that still holds its
   prefix unless that GPU is busy; otherwise to the least-loaded GPU, and
   prefetch its prefix from the pool to that GPU while it waits. Prediction
   helps here too: a session predicted to return within seconds should be
   kept routable to its GPU; one predicted to wait minutes can be rebalanced
   freely.
2. **Pool eviction.** Keep in the shared CPU pool the sessions most likely to
   return before they would be evicted anyway, weighted by the disk reads an
   eviction would cause. This is our current policy plus the fixes below.

The co-design claim to test: prediction-aware routing plus pool eviction
beats either alone, and beats prefix-aware routing that ignores agent timing.

## Policy fixes to fold in

- **Recency guard.** Do not rank a session by prediction until it has been
  idle a few seconds. Mispredicted quick returns were the policy's worst
  losses (+39% to +65% p90 when CPU served quick returns).
- **Probability of returning in time.** Rank by P(return before the pool
  would evict it anyway | time already waited), from per-tool wait
  distributions, instead of one predicted return time.
- **Cost per byte.** Weight by the disk reads an eviction would force and how
  busy the disk link is, since that is where the gains come from.

## Baselines

- LRU pool, ARC pool (built into vLLM).
- Round-robin and least-loaded routing (no affinity).
- Prefix/cache-aware routing (SGLang router, llm-d KV-aware routing, Preble).
- Agent-aware GPU-only management (CacheWise, Continuum) on each GPU.
- Idleness-based placement (MORI).

## Metrics

p50/p90/p99 resume TTFT, disk read bytes, pool hit rate, GPU utilization and
queueing (to show load is sustainable), and fairness across sessions. Always
report the load and the affinity next to each result, with 5-sample means and
spread.

## Experiments

1. **Storage speed (running).** Rerun the winning cells with 3 and 7 GB/s
   disks. If the gain vanishes with fast storage, the story narrows to slow or
   remote shared storage.
2. **Policy fixes.** Recency guard and probability ranking, then rerun the
   multi-GPU sweep.
3. **Routing.** Add prediction-aware routing to the simulator; compare the
   routing x pool-policy grid.
4. **Scale.** 2, 4, 8 GPUs; pool size relative to GPU memory; load relative to
   capacity.
5. **Live check.** One multi-GPU run on Modal (e.g. 2 GPUs, shared CPU pool
   via LMCache or a single host with two GPUs), detached, against the
   simulator's prediction.

## Literature check (Oct 2026)

Closest work, grouped. Several entries are 2026 preprints; numbers are as
reported by their authors.

**Agent serving across replicas**
- **ThunderAgent** (arXiv 2602.13692; GT, UIUC, CMU, Together AI). Global
  program-aware queue over data-parallel replicas; a paused program can resume
  on any replica. It deliberately drops sticky routing, assumes a paused
  program's KV is evicted (no shared pool; offload treated as orthogonal), and
  argues tool durations are unpredictable, using elapsed-time decay instead.
  Reports 1.5-3.6x serving throughput over vLLM and 2.4x over SGLang
  cache-aware routing at 88 H100s. Adopted in SkyRL and NVIDIA Dynamo. The
  strongest competitor and the opposite design choice to ours.
- **MISA-T** (arXiv 2608.11152; Nanjing Univ., StepFun). RL rollouts over a
  shared inference pool: caps how many sessions compete for KV, and holds a
  request for its cached instance when the wait plus hit beats routing it cold
  (their Eq. 7). Uses queue and prefill timing and per-class residency, not
  per-session tool duration. +44-53% rollout throughput over vLLM Router.
  Closest to our "hold vs rebalance" idea.
- **MORI** (arXiv 2606.00866). Idleness-ranked GPU/CPU placement, also across
  replicas (DP=3); SSD only as future work.

**Agent-aware eviction on one engine**
- **CacheScout** (arXiv 2608.14624; UCSC, UW, UChicago/LMCache). Learns which
  agent runs next and keeps its prefix by survival probability, recency and
  rebuild cost, with LMCache CPU offload. -18-45% TTFT over vLLM. Very close to
  our probability-ranking fix, but single engine and multi-agent workflows,
  not tool timing.
- **Continuum** (TTL pinning across tool calls), **CacheWise**, **TokenCake**,
  **KVFlow**: single engine.

**KV-aware routing (not agent-aware)**
- **CacheRoute** (arXiv 2608.19677, Meta): planned prefix-affinity routing,
  93% hit vs 64% for flat load balancing, 2.3x capacity at a p99 SLO. Notes an
  agent id could be the routing key; not evaluated.
- Preble, DualMap, CHWBL, llm-d, SGLang router, Ray Serve and GKE prefix-aware
  routing. **PEEK** (arXiv 2607.02525): queue-aware eviction within a replica.
- **Unified AI Gateway** (arXiv 2609.06940, Huawei): joint model routing and KV
  actions, analytic only.

**Eviction-policy studies (support our negative result)**
- **When Fancy Eviction Fails** (arXiv 2609.28870, Harvard): 14 policies on
  agentic production traces, per-replica HBM and a global DRAM/SSD pool. At
  large capacity LRU is within 0.4 points of the best online policy; gains
  only at small capacity. No agent-timing policy tested.
- **Where Should the KV Cache Live?** (arXiv 2609.16215): GPU/CPU/SSD
  placement in a simulator without admission; predicted reuse matches
  recency; prefetch does not pay on a contended link (as we found).

**What is still open**
- Agent-aware eviction in a KV pool shared by several replicas: none found.
- Routing that uses predicted tool duration to decide hold vs rebalance:
  none. ThunderAgent rejects prediction; MISA-T uses queue timing.
- Any of the above evaluated with GPU admission modeled and a sustainable
  load. Most simulator papers here do not model admission.

**Implications**
- The pitch must be narrow: predicted return time for hold-vs-move routing
  plus shared-pool eviction, and when it beats ThunderAgent-style "resume
  anywhere, recompute" and MISA-T-style hold-or-cold routing.
- ThunderAgent is the baseline to beat. Its premise (paused KV is gone) is
  exactly what a shared pool changes; our slow-storage result suggests the
  pool only matters when reloads are expensive.
- Our negative single-server result agrees with "When Fancy Eviction Fails",
  and adds the reason: with GPU admission and a sustainable load, the GPU cache
  already holds what returns soon.

## Risks

- **Novelty.** KV-aware routing exists (SGLang router, llm-d, Mooncake's
  Conductor, Preble). Our angle must be the agent timing signal in routing
  and pool eviction; this needs a literature check before committing.
- **Storage dependence.** If fast NVMe or RDMA storage removes the disk
  bottleneck, gains may only exist for slow or remote storage.
- **Edge-of-capacity effect.** Gains so far appear near saturation with high
  variance; a reviewer will ask whether operators run there.
- **Simulator fidelity.** Disk thrash and batched decode are not modeled yet;
  the live check is what makes the numbers credible.
