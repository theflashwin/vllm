# Novelty audit of the proposed KV placement project

Reviewed October 4, 2026. This is a scoped literature assessment, not a proof
that no related work exists. The proposal reviewed is *Workload and Topology
Aware KV Cache Migration on vLLM*, supplied by the project authors.

## Verdict

The broad mechanism as proposed is already covered by prior work. The plan
does not currently identify a defensible research distinction. Its strongest
overlaps are its own references 8 and 9. Implementing and evaluating these
ideas in native vLLM could still make a useful systems project; implementation
location alone does not establish a new placement algorithm.

Novelty and performance are separate questions. Our modest simulation gains
do not establish lack of novelty, and prior overlap does not imply that a
carefully designed new mechanism could not improve performance.

## Direct overlap with the original proposal

| Proposed element | Closest evidence | Assessment |
| --- | --- | --- |
| Choose GPU, CPU, or storage for each block | Placement paper, III-A/B; multi-tier paper, III-B | Already studied |
| Predict reuse from workload metadata | Multi-tier paper, III-C/G; CacheWise, 5.2 | Already studied |
| Promote state before reuse | Placement paper, III-C; multi-tier paper, III-B/E/G | Already studied |
| Use tier latency/bandwidth to guide decisions | Placement paper, III-A/D; multi-tier paper, III-B | A bandwidth table alone is insufficient differentiation |
| Integrate into vLLM and replay real agent traces | CacheWise, 5.3/6.1; our trace experiment | Useful implementation/evaluation contribution; not inherently a new mechanism |

### Reference 8: GPU/CPU/SSD placement

[Where Should the KV Cache Live?](https://arxiv.org/html/2609.16215v1)
explicitly studies block placement, promotion, eviction, and predictive
prefetch across the same three tiers. Sections III and V-E are the closest
comparison. It also charges speculative and demand traffic to a shared link.
Its evaluation uses a single-GPU, batch-one simulation with synthetic
workloads; its large capacity gains follow from added tier capacity, rather
than a superior policy. Section VI limits the generality of its results.
Real hardware and real traces would strengthen the evidence, but repeating
the general placement question would need to be presented as an extension
or validation study.

### Reference 9: predictive multi-tier management

[Predictive Multi-Tier Memory Management](https://arxiv.org/html/2604.26968v2)
describes latency-aware placement, reuse-dependent tier thresholds, and
asynchronous promotion/demotion in III-B. III-C uses Bayesian reuse estimates
and confidence weighting; III-G considers agent task transitions. This
directly overlaps the proposed combination of workload prediction and tier
costs. Its cluster speedups are analytical projections from component trace
validation and hardware specifications, not measured end-to-end cluster
results. That limits the performance evidence; it does not make the broad
mechanism new.

### Reference 5: agent reuse prediction in vLLM

[CacheWise](https://arxiv.org/html/2606.16824v1), sections 5.2–6.1, predicts
remaining tool time conditioned on elapsed time and tool metadata, then uses
the estimates for eviction. Tool arguments and clustering are already part
of its predictor. It also integrates prefix-aware scheduling into vLLM and
evaluates real coding-agent traces. A richer tool-name predictor or a vLLM
implementation is consequently not enough to distinguish this project.

## What the plan must change

The problem statement overstates the gap when it suggests that existing work
principally addresses when state leaves GPU memory rather than where it goes.
The state-of-the-art section already acknowledges predictive placement;
the solution must explain exactly how its decision rule differs from it.

“Topology awareness” needs an operational definition. Fixed per-tier transfer
rates describe a cost model. A stronger possible hypothesis would involve
online decisions under changing shared-link contention, transfer dependencies,
and demand deadlines. This is a candidate to investigate, not an established
novel contribution.

Before building another controller:

1. Specify its state, decisions, objective, and constraints. State precisely
   which prior algorithm cannot make the same decision.
2. Run an oracle headroom experiment with paired request traffic and the same
   resource accounting. Measure time on the resumption critical path, not only
   cache hits or speculative bytes.
3. Compare against the closest reuse-aware placement policy and a simple
   bandwidth budget, in addition to LRU/ARC.
4. Validate costs and overlap on a GPU. Our current simulator omits GPU/CPU
   contention, write contention, active working-set pinning, and decode
   scheduling effects.
5. Choose a minimum useful latency gain and a tail-regression limit before
   tuning. Stop if even the oracle cannot meet them.

## Directions that should not be claimed as new without further evidence

The broader literature screen found close work on reload-versus-recompute
([CacheFlow](https://arxiv.org/abs/2604.25080)), duration-distribution retention
([Continuum](https://arxiv.org/abs/2511.02230)), queued-prefix protection and
scheduling ([PEEK](https://arxiv.org/abs/2607.02525)), dynamic workflow prediction
([PBKV](https://arxiv.org/abs/2605.06472)), online agent-transition learning
([CacheScout](https://arxiv.org/abs/2608.14624)), and joint DAG scheduling and
retention ([TOPAS](https://arxiv.org/abs/2608.25523)). These references rule out
casually presenting the corresponding broad ideas as unexplored. This audit
does not certify a new gap against every implementation in that set.

## Recommendation

Keep the project if the goal is an eight-week implementation and empirical
study. Describe it as testing whether reuse-aware placement improves native
vLLM under measured hardware costs and real agent traces. For a novel research
paper, revise the proposal around a specific uncovered mechanism and establish
headroom before expanding the implementation. No novel mechanism has yet
been established by this review.

## Follow-up screen and GPU evidence

The follow-up search also found
[MORI](https://arxiv.org/abs/2606.00866), which ranks agent programs by relative
idleness and adapts the GPU/CPU partition to hardware capacity, and
[EfficientAgent](https://arxiv.org/abs/2609.33762), which estimates the host
reuse working set and changes write admission when the tier cannot hold it.
Generic activity-aware placement or selective writes should therefore not
be presented as new without a more specific distinction.

[Mooncake's TENT documentation](https://kvcache-ai.github.io/Mooncake/design/tent/deadline-scheduling.html)
already covers deadline ordering and infeasible-transfer admission. Its
documented feasibility logic is specific to RDMA, with staged-transfer and
other-transport limitations. Joint decisions across a staged path remain a
candidate extension to examine, not a certified unexplored problem.

The first concrete headroom test is recorded in
[RESTORATION_RESULTS.md](RESTORATION_RESULTS.md). On an L4 with Qwen2.5-1.5B,
full reload wins over recomputation and three fixed load caps, with and
without competing H2D traffic. This does not motivate an adaptive controller
for the tested conditions. Secondary-storage contention and mid-transfer
replanning remain unvalidated.
