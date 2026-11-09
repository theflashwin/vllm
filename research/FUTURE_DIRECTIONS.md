# Future directions, with literature review (Nov 2026)

Seven candidate directions, each checked against the literature (arXiv,
GitHub, vendor docs; 2024 to Nov 2026). Verdicts: **open**, **partly done**
(a specific gap remains), **done**. Several sources are 2026 preprints read
from abstracts or partial text; those are marked "(abstract)".

## Where we stand

- Agent-aware CPU-tier eviction does not lower TTFT at sustainable load on one
  server: the GPU cache already holds sessions that return soon, and the rest
  have left CPU too. Live A10G (stable): ours +56% p90 vs LRU.
- When disk traffic is heavy, ours does cut it (live A100: 35% fewer disk
  reads, 23-28% fewer recomputed tokens), but latency is set by GPU admission
  (requests waiting for memory), so TTFT is unchanged.
- The only simulated latency win: several GPUs sharing a slow (~1 GB/s) disk
  near saturation; gone at 3-7 GB/s.
- Prerequisite for any simulator-based direction: fix the audit's bugs (one
  GPU memory pool for cache and running requests, shared-prefix inflation,
  session splitting on tiny context shrinks, warm-up timing, paired routing
  [done], recency guard [done]) and tag results with the code version.

## Summary

| | Direction | Verdict | Recommendation |
|---|---|---|---|
| A | Re-evaluate published agent KV-cache claims under GPU admission and sustainable load | Partly done; the specific angle is open | **Pursue (main)** |
| C | Prompt-cache retention economics for agents | Partly done; per-call tier choice and provider cost model open | **Pursue (second track)** |
| G | Efficiency, not latency: what eviction saves when latency is admission-bound | Partly done; policy-vs-LRU efficiency and capacity at SLO open | Pursue as part of A, with a go/no-go |
| B | "KV pressure index": when tiering/eviction can matter | Mostly done | One explanatory figure, not a project |
| D | When does wait prediction pay off? | Mostly done | One axis inside A |
| E | Predicted-return routing + shared-pool eviction across GPUs | Mostly done | Drop (narrow, incremental on MORI) |
| F | Admission-aware keep/drop/offload per tool call | Done in pieces | Drop (evaluation-only) |

---

## A. Re-evaluating agent KV-cache claims under admission and sustainable load

**Idea.** Port published agent-aware policies into our simulator, sweep load
as a fraction of the sustainable rate (λ/λ\*) with GPU admission on and off,
with storage tiers and replicas, and check one or two live. Add a "pitfall
ledger": which conclusions flip under overload, no admission, timestamp
inflation, hint-only requests, closed- vs open-loop replay.

**Closest work.**
- [AgentServeSim](https://arxiv.org/abs/2606.09613) (UCF, Sep 2026): ports
  evict-always, LRU, fixed TTL, SAGA, Continuum-style per-tool TTL, InferCept;
  finds retention alone moves JCT ~+0.4%. No storage tiers, no admission
  toggle, load not normalized to capacity, no code.
- [When Fancy Eviction Fails](https://arxiv.org/abs/2609.28870) (Harvard, Sep
  2026): 14 generic policies, LRU near-optimal; explicitly excludes
  tool-time-aware policies; fixed concurrency.
- [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849) (Tsinghua,
  Alibaba, Sep 2026): live comparison of Continuum, CacheWise, TokenCake,
  InferCept predictors vs LRU and oracle; predictors can lose to LRU; no load
  sweep, no admission.
- [MARS](https://arxiv.org/abs/2604.26963): the only system paper stating
  gains depend on load.
- No independent reproduction of any of these systems was found; every
  cross-comparison is by a competing author.

**Open angle.** Capacity-normalized load + admission toggle + storage tiers +
replicas, applied to the agent-aware policies the other studies left out, plus
measured pitfall effects.

**Public code.** Continuum ([vllm-continuum](https://github.com/Hanchenli/vllm-continuum),
note: no predictor, fixed-TTL only), [InferCept](https://github.com/WukLab/InferCept),
[KVFlow](https://github.com/PanZaifeng/KVFlow) (SGLang),
[TokenCake](https://github.com/pku-lemonade/TokenCake),
[ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent),
[MARS](https://github.com/Afterglow231/MARS_preview). Not public: CacheWise
(traces only), CacheScout, MORI, AgentServeSim.

**First experiment.** Measure λ\* per GPU (L4/A10G/A100) at a p90 SLO; sweep
λ/λ\* ∈ {0.3, 0.5, 0.7, 0.85, 1.0, 1.2}, admission on/off; ~6 ported policies
(Continuum TTL, SAGA, InferCept min-waste, ThunderAgent decay, MORI idleness,
CacheWise remaining-time) vs LRU on TraceLab and CATraces; pitfall ledger.
Live: Continuum fork vs stock vLLM on A10G at one point below and one above
λ\*. Cost: $0 sim, ~$5-15 live.

**Expected signal.** Published-scale gains when overloaded or without
admission, shrinking to ~0 at ≤0.85 λ\* with admission. Matches our results,
AgentServeSim and Fancy Eviction.

**Risks.** Ports called strawmen (validate one against its repo); a negative
result (frame as "where each claim holds"); operators run near saturation
(report jobs/hour/GPU at the SLO).

## C. Prompt-cache retention economics for agents

**Idea.** (a) Client side: per call, choose a provider cache tier (Anthropic
5-min write 1.25x / 1-hour 2x / reads 0.1x; OpenAI 30m 1.25x / 12h 2.5x on
newer models; Gemini storage per token-hour) or keepalive pings or expiry,
using per-tool and user-wait predictions, on real Claude Code traces ("ski
rental with predictions"). (b) Provider side: holding cost in HBM / DRAM / SSD
vs recompute from our measured costs, compared with price menus and margins.
(c) Systems piece: simulate the fleet effect of widespread keepalives and
per-call tiers on cache pressure.

**Closest work.**
- [Keeping the Cache Warm Pays](https://arxiv.org/abs/2607.19214) (Khailo, Jul
  2026): fixed-timer keepalive economics and break-even idle times per
  provider, synthetic probes; no real traces, no per-call tier choice, no
  provider cost model; fleet externality argued, not simulated.
- [Don't Break the Cache](https://arxiv.org/abs/2601.06007): where to put
  cache breakpoints; not TTL choice.
- [Can I Buy Your KV Cache?](https://arxiv.org/abs/2606.13361): reuse 9-50x
  cheaper than re-prefill vs a 0.1x read price; no holding cost over time.
- [TraceLab](https://arxiv.org/abs/2606.30560): says retention break-even is
  "not quantified".
- Practitioner tools ping on fixed timers (aider, claude-code-cache-keepalive);
  none predict.

**Open angle.** Per-call tier/keepalive choice from predicted waits on real
traces, provider-side cost-vs-price analysis, simulated fleet externality.

**First experiment.** From TraceLab/CATraces, gaps by tool and user turns
(share <5 min, 5-60 min, >60 min); replay prefix lengths under always-5-min,
always-1-hour, fixed keepalive, per-tool predictor, oracle at current prices;
dollars per session and predictor-oracle gap. Settle with a few dollars of
live probes whether a post-response 1-hour ping bills as a read or a
re-write (decides whether prediction can use the response's stop reason).
Cost: $0 analysis, <$10 probes.

**Expected signal.** Small on tool-to-tool gaps (97.5% hits already), larger
on user-turn returns.

**Risks.** Reads as economics unless the fleet simulation is strong; prices
and semantics change monthly (date-stamp everything); provider TTLs are
best-effort; must differentiate clearly from Khailo.

## G. Efficiency, not latency

**Idea.** When latency is admission-bound, express what agent-aware eviction
saves vs LRU as GPU-seconds, energy, disk I/O, dollars, and especially
capacity: max sessions per GPU at a p90 SLO (less recompute shortens how long
requests hold memory, so admission-bound capacity could rise even if TTFT at
a fixed load does not).

**Closest work.**
- [Mao & Zhou, Energy Characterization of KV Cache Offloading Under Agentic
  Workloads](https://hotcarbon.org/assets/2026/paper-20.pdf) (Georgia Tech,
  HotCarbon, Jul 2026): offload on vs off; 14-32% GPU energy saved above ~90%
  KV utilization, within noise below; no eviction policy comparison, no disk,
  no capacity at SLO. Same school; worth contacting.
- [EfficientAgent](https://arxiv.org/abs/2609.33762) (HKUST/Huawei, Sep 2026):
  when offloading helps; no energy, no TTFT.
- SSD wear: [Gandalf](https://arxiv.org/abs/2610.30791),
  [SARIA](https://arxiv.org/abs/2610.31169) (uses TraceLab) (abstracts).
- Agent power: [KAIROS](https://arxiv.org/abs/2604.16682),
  [Where Does the Energy Go?](https://arxiv.org/abs/2609.29707) (abstract).

**Open angle.** Eviction policy vs LRU in the same tiered system, efficiency
and capacity-at-SLO metrics, multi-GPU shared disk bandwidth.

**First experiment.** $0: convert the live A100 savings to GPU-seconds and
joules; simulate max sessions/GPU at a p90 SLO for LRU vs ours. ~$3: NVML
energy for recompute vs reload of 4k/16k/64k prefixes on A100 (verify NVML
access on Modal).

**Go/no-go.** If capacity gain is <5% and energy is within the ±3% noise
floor, drop it.

**Risks.** Weak signal (rough estimate ~7% energy); "fewer recomputed tokens
means less energy" is near-tautological without the capacity result.

## B. KV pressure index (mostly done)

**Idea.** Idle KV during waits (Little's law) vs GPU memory left after running
requests, predicting when tiers/eviction matter.

**Closest work.** [EfficientAgent](https://arxiv.org/abs/2609.33762)
(pre-registered predictor of when offloading helps), Mao & Zhou (empirical
~90% utilization threshold), [Fancy Eviction](https://arxiv.org/abs/2609.28870)
(Little's-law working set), [Yao et al.](https://arxiv.org/abs/2609.02027)
(closed-form multi-turn LRU hit ratio),
[Nie et al.](https://arxiv.org/abs/2605.04595) (KV-constrained stability,
ICML 2026), [KVSET](https://arxiv.org/abs/2609.27746) (abstract).

**Remaining gap.** One index combining running-request memory with idle KV,
predicting latency rather than hit rate. **Use:** one figure in A or G that
explains "hit rate improves, TTFT doesn't".

## D. When does wait prediction pay off? (mostly done)

**Closest work.** [Ask the Tool](https://arxiv.org/abs/2609.18849) (live
predictor comparison), a public Stony Brook class project
([agent-kv-retention](https://github.com/Omkar38/agent-kv-retention):
oracle/break-even/Continuum-TTL/learning-augmented, 19 prediction-error
conditions, 1-2% policy differences), CacheWise\* (ground truth ≈ predictor).
Note: LAMPS uses prediction; MARS is the observation-only system;
ConServe is about P/D disaggregation.

**Use.** Run each policy in A in oracle, predicted and observation-only
variants with injected error; cite both as concurrent work.

## E. Predicted-return routing and shared-pool eviction (mostly done)

**Closest work.** [MORI](https://arxiv.org/abs/2606.00866) (return-to-holder
routing, sticky placement, idleness ranking, DP=3; no disk or shared pool),
[vLLM AgentX blog](https://vllm.ai/blog/2026-09-08-vllm-agentx) (session-sticky
routing wins even with a shared Mooncake pool; lists tool-latency hints as
future work), [SAGA](https://arxiv.org/abs/2605.00528) (affinity batching,
work stealing, KV migration on 64 GPUs),
[ThunderAgent](https://arxiv.org/abs/2602.13692),
[MISA-T](https://arxiv.org/abs/2608.11152),
[CacheRoute](https://arxiv.org/abs/2608.19677). A community reproduction on
393 Claude Code sessions found no return-aware policy beat LRU when
capacity-bound ([agentic-kv-cache](https://github.com/gauravapiscean/agentic-kv-cache)).

**Remaining gap.** Predicted-return eviction in a shared slow remote pool +
hold-or-move by predicted wait. Narrow, increment on MORI; our own wins exist
only near saturation of a ~1 GB/s link.

## F. Admission-aware KV retention (done in pieces)

**Closest work.** [InferCept](https://arxiv.org/abs/2402.01869) (per-call
preserve/discard/swap, elapsed-time estimate ≈93% of oracle),
[Continuum](https://arxiv.org/abs/2511.02230) (per-tool TTL coupled to
queueing, evicts pinned KV to admit waiting requests),
[MORI](https://arxiv.org/abs/2606.00866) (tiered admission),
[Ask the Tool](https://arxiv.org/abs/2609.18849) (tool progress signals),
[TokenCake](https://arxiv.org/abs/2510.18586),
[LAMPS](https://arxiv.org/abs/2410.18248).

**Remaining gap.** Combining these; only defensible as an evaluation, which
is A.

---

## Recommendation

**Main project: A**, "Do agent-aware KV-cache claims survive GPU admission and
sustainable load?"
- Uses our strongest assets: the admission finding, a live-validated
  simulator (after the audit fixes), a live harness, two converted traces.
- Fold in D (oracle / predicted / observation-only variants with injected
  error), B (one explanatory figure), and G's capacity-at-SLO metric
  (jobs/hour/GPU), which also answers "operators run near saturation".
- Differentiate from AgentServeSim (normalized load, admission toggle, tiers,
  replicas), Fancy Eviction (agent-aware policies), Ask the Tool and the Stony
  Brook project (cite as concurrent; do not claim the predictor question).

**Second track (one person): C**, retention economics: per-call tier and
keepalive choice from predicted waits, provider cost-vs-price, simulated fleet
externality. Cheap, least crowded, uses the bimodal-wait finding directly.

**Budget.** A: ~$5-15 live. C: <$10 probes. G microbenchmark: ~$3. Total under
$25.

**Order.** (1) Fix the simulator bugs and tag results; (2) measure λ\* and
rerun our own policy as the first "re-evaluated claim"; (3) port policies;
(4) live checks; C in parallel from week 1.

---

## Second round (Nov 2026): beyond eviction and retention

Motivated by two measurements: latency is set by GPU admission, and the disk
tier barely helps latency at sustainable load. Gain figures below are rough:
reviewers' quick simulations (one or few seeds, not our fixed simulator) or
authors' own reports. "Open" means no prior work was found, not that none exists.

| | Direction | Verdict | Plausible win | Main risk |
|---|---|---|---|---|
| H | **Context as a load-driven capacity knob** | Open | 1.2-2x capacity at SLO when memory-bound | Task quality evaluation on budget; a fixed threshold may do as well |
| I | Whole-task overload control: what to shed or degrade | Partly done (pausing is done) | Waste 37% to <10% is a target, not yet shown | No trace records abandonment; pausing may capture most of it |
| J | Attained-service preemption with fast swap | Partly done | Turn p50/p90 x2-18 by load and GPU; capacity only +3-10% | "FastServe for agents" |
| K | Speculative decoding to cut KV scans | Open as framed | x1.1-2.3 on low-bandwidth GPUs; can lose on A100 | Acceptance rate unknown; kernel may not reuse the scan |
| L | Load-adaptive KV precision | Static done; dynamic open | x1.4-4 measured elsewhere (static) | Kernels and accuracy evaluation |
| M | Rehoming idle sessions so scaling takes effect fast | Open for agents | 5-15% capacity (a guess) | Replica start time dominates; AgentX found sticky routing wins |
| N | Failover SLOs for idle sessions | Idle angle open | ~2x shorter transient | Modest effect; second evaluation at best |

**Dropped:** spot recovery from a durable tier (under 1 point of cost at realistic
rates), predictive autoscaling from tool state, KV sharing between running
requests (vLLM already does it; x1.04-1.09), lossless paging over PCIe (19-49x
slower than scanning), tiny-turn fast paths, tokenization, streaming parsers,
forks.

### H. Context as a load-driven capacity knob

**Idea.** Clients choose when to compact, ignoring server load. Let the server use
memory pressure to ask the largest sessions to drop old tool results or compact.

**Why it might work.** Each decode step reads a request's whole KV, so memory and
memory bandwidth both scale with context: capacity ~ BW / (KV bytes x context x
output steps). Context is highly concentrated: in full TraceLab, turns over 200k
tokens are 40% of turns and 75% of context sum, from 10% of sessions (measured
before the converter fix; recheck).

**Closest work.** [AKV](https://arxiv.org/html/2610.28577v1) (Oct 2026): fixed rolling
tool-result drop, 1.18x throughput, 27.7% lower cost, pass rate +1.7 points; not
load-driven. [Aries](https://arxiv.org/pdf/2607.29069) (Aug 2026): names this control
plane as future work (scoop risk). [TokenPilot](https://arxiv.org/html/2606.17016v1),
[OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction): client
thresholds, no load signal.

**First experiments.** (1) ~$2: decode step time vs concurrency {1, 2, 4} and context
{8k, 32k, 80k} to validate the scan model, which is unvalidated at long context and
may also explain the simulator's 2-2.5x gap. (2) $0: add the scan model and context
limits (uniform, tail-first, pressure-driven) to the simulator; measure max sessions
at a p90 SLO. **Drop if under 15% capacity**, or if a fixed low threshold matches
adaptive and tail-first adds nothing. (3) $5-15: live proxy reading KV usage and
queue length. (4) Quality pilot on 20-30 SWE-style tasks, plus published curves.
