# Budgeted, uncertainty-aware KV prefetch

## Question and decision

Can secondary-to-CPU prefetch lower resumption latency while spending fewer
bytes and limiting interference with foreground reads? The earlier paired
TraceLab experiment found modest latency gains and substantial prefetch waste.
Those results used instant prefetch insertion and uncontended transfers.

This experiment is implemented in `kvplace.prefetch_experiment`. It changes
the secondary-read model for **every** variant. It does not implement a live
vLLM prefetch hook. Absolute latency numbers cannot be compared directly with
the earlier simulator table.

## Controlled comparison

All variants use CPU LRU, write-through secondary storage, the online tool
predictor, and one hint request per turn. Each run starts with empty caches
and empty predictor history. Session starts are fixed; later arrivals follow
response completion plus the recorded tool gap, preserving the agent's
dependency on earlier responses.

| Variant | Timing | Admission |
|---|---|---|
| `demand` | No prefetch | Demand loads only |
| `fixed` | Tool median minus transfer time and 0.5 s | Entire missing prefix; can queue behind demand |
| `budget` | Tool median minus chunk transfer time and 0.5 s | Idle link, rate budget, occupancy budget, bounded chunks |
| `uncertain` | Same median quantile timing | Budget rules plus conditional probability gate |

The predictor uses a rolling history of 200 completed calls per tool. The
history is snapshotted when the hint finishes delivery. Neither admission nor
timing reads the current call's recorded duration. Completion labels used for
probability evaluation are added only after replay.

For elapsed tool time `e`, chunk transfer time `d`, and residency window `w`,
the gate estimates `P(e < duration <= e + d + w | duration > e)` by counting
historical observations. Require at least eight observations before enabling
the uncertainty variant, and require estimated probability at least 0.5.
This is an empirical estimate, not a calibrated confidence guarantee.

The initial parameters are fixed before the finalized runs:

- Chunk ceiling and token-bucket burst: 256 blocks (about 235 MB).
- Token-bucket refill: 10% of the configured secondary-read bandwidth.
- Resident plus in-flight prefetch cap: 10% of CPU blocks.
- Residency window: 2 seconds; scheduling quantile: 0.5.
- Reject admission when the shared read link is busy. Recheck after each
  completed chunk; rejected jobs are not retried later in the same turn.

Transferred chunks become CPU-resident only when their completion event
fires. Reads serialize on one nonpreemptible secondary-read link. Demand
reads can wait behind an already admitted chunk. Stale or redundant
completions consume bandwidth and count as waste. Prefetch follows the
missing prefix in order; it does not skip an in-flight missing prefix block.
The occupancy cap bounds speculative data; it does not pin demand blocks or
guarantee that their eviction is harmless.

## Reproduce

Generate the full TraceLab v0.0.2 Claude subset with the commands in README:
300 sessions, 8,081 turns, seed 0, arrival rate 0.05/s, token scale 1, maximum
context 200,000. Input gzip SHA-256:
`11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65`.

```bash
export PYTHONPATH=$PWD/research
.venv/bin/python -m kvplace.prefetch_experiment \
  research/traces/tl300_full_r05.jsonl --gpu-blocks 12000 --cpu-blocks 24000 \
  --out /tmp/prefetch_full.json
.venv/bin/python -m kvplace.prefetch_experiment \
  research/traces/tl300_full_r05.jsonl --gpu-blocks 12000 --cpu-blocks 24000 \
  --arrival-scale 5 --out /tmp/prefetch_light.json
.venv/bin/python -m pytest research/tests/test_sim.py research/tests/test_tracelab.py \
  -q -p no:cacheprovider
```

The second run changes only session start times: arrival rate 0.01/s, with
unchanged tool gaps, tokens, and sessions. This tests sensitivity to offered
load. Both runs use uncalibrated CostModel defaults, including 3 GB/s for
secondary reads. JSON outputs include exact configuration, per-turn records,
summaries, and probability decisions with retrospective labels.

Report p50/p90/p99 resumption TTFT, demand and prefetch bytes, useful prefetch
fraction, stale bytes, read-link wait, rejection reasons, and peak speculative
occupancy. Check `used + wasted == transferred` for every run. Brier score
describes predictions on evaluated gate decisions, not a held-out workload.

## Validation and next gate

### Initial results

One deterministic simulation per variant at each arrival scale; no GPU runs.
Exact costs, configuration, and counters are in
[prefetch_results.json](prefetch_results.json). GB below means decimal GB.

| Session arrivals/s | Variant | TTFT p50 (s) | p90 (s) | p99 (s) | Prefetch GB | Useful prefetch |
|---|---|---:|---:|---:|---:|---:|
| 0.05 | demand | 0.251 | 170.445 | 537.775 | 0 | n/a |
| 0.05 | fixed | 0.251 | 173.449 | 586.315 | 1,846.5 | 12.6% |
| 0.05 | budget | 0.254 | 145.252 | 581.642 | 18.9 | 66.5% |
| 0.05 | uncertain | 0.251 | 170.445 | 537.775 | 0 | n/a |
| 0.01 | demand | 0.161 | 2.403 | 8.468 | 0 | n/a |
| 0.01 | fixed | 0.160 | 2.388 | 8.669 | 2,174.1 | 28.9% |
| 0.01 | budget | 0.159 | 2.348 | 8.436 | 61.6 | 41.1% |
| 0.01 | uncertain | 0.161 | 2.396 | 8.472 | 7.4 | 61.5% |

- Budget-only cuts speculative bytes by 97–99% relative to fixed prefetch.
  At 0.01 arrivals/s, p90 improves 2.3% versus demand and p99 improves 0.4%.
- At 0.05 arrivals/s, budget-only improves p90 14.8% but worsens p99 8.2%.
  This is not a consistent tail-latency win. The absolute latencies expose
  heavy queueing under the uncalibrated 3 GB/s read-link assumption.
- At 0.01 arrivals/s, the uncertainty gate reduces budget-only prefetch bytes
  another 88%, but its p90 benefit versus demand is only 0.3%. At 0.05/s it
  admits no prefetch and exactly matches demand-only.
- Fewer speculative bytes do not imply fewer total reads than demand-only.
  At 0.01/s, total secondary reads are 17.759 TB for demand, 20.266 TB for
  fixed, 17.841 TB for budget, and 17.900 TB for uncertain. Cache state and
  later request arrival times change with each policy.
- The lower-rate gate admitted 78 chunk decisions: mean predicted return
  probability was 51.5%, versus 43.6% observed returns within the window.
  This exploratory estimate is overconfident on that selected group; useful
  block fraction measures a different outcome from return timing.
- Across all eight runs, transferred blocks equal useful plus wasted blocks,
  hint-request counts equal 8,081, and bounded variants obey the 2,400-block
  resident-plus-in-flight cap.

**Decision:** retain budget-only as the strongest initial candidate for
calibrated evaluation. This uncertainty gate improves selectivity but has not
earned a live latency-optimization implementation. Do not tune its threshold
on this sample and treat the resulting gain as independent validation.

Tests exercise the budget through full simulation, gate rejection with cold
history and a narrow window, and late prefetch completion with demand waiting.
The cheapest coverage is simulator unit tests: this change has no live engine
or model-output path.

A promising result must improve on demand-only and demonstrate what the
probability gate adds beyond budget-only. Lower waste alone does not justify
claiming a serving speedup. These runs are exploratory and use one real trace
sample; they do not establish novelty or generalization.

Before a live implementation, measure the secondary-read path and calibrate
the model. Then validate an independent trace sample and repeat GPU runs with
identical hint transport, measuring p99 TTFT, task completion, throughput,
migration bytes, and generation correctness. Require a repeatable latency
benefit without a material throughput regression.

Remaining simulator limitations: demand cache updates happen at arrival,
GPU working sets are not pinned, GPU/CPU transfers and secondary writes do not
contend with reads, prefill is FIFO, and decode has no shared compute cost.
The probability window is a heuristic, not an enforced residency TTL.

Related work to distinguish before a paper or upstream proposal:
[Continuum](https://arxiv.org/abs/2511.02230) studies retention and scheduling
during tool waits; [PEEK](https://arxiv.org/abs/2607.02525) combines queue-aware
scheduling and eviction. Existing
[vLLM programmable-cache proposals](https://github.com/vllm-project/vllm/issues/57103)
cover lifecycle and movement contracts. The proposed research contribution is
uncertain return timing under explicit transfer and speculative-memory
budgets; its novelty remains to be checked against the full literature.

AI assistance: Codex implemented this exploratory simulator and plan. A human
must review the changes and validate serving behavior before an upstream PR.
