# Restoration under contention: GPU headroom experiment

## Question

Does changing the fraction of an offloaded prefix that is loaded versus
recomputed offer meaningful resumption-latency headroom as transfer contention
changes? This experiment establishes headroom before building an online
controller. It does not establish novelty.

## First run on Modal

- One CUDA GPU; proposed initial device: L4.
- One small dense model, BF16, tensor parallel size 1.
- Fixed engine configuration and batch size 1 for the initial crossover test.
- Bounded one-shot function with a timeout, no deployed endpoint, no retries.
- User spending limit: $1 for the initial probe. Modal function
  timeouts bound execution time, not the entire workspace's dollar spend.
- Record GPU name, PCIe topology where exposed, CUDA/PyTorch/vLLM versions,
  source revision, model revision, and all experiment parameters.

An L4 result is specific to that device's compute-to-transfer ratio. Confirm
a promising crossover on a second GPU before making a general hardware claim.

## Controlled arms

Use the native OffloadingConnector's per-request `max_load_tokens` control:

| Arm | External prefix loading |
| --- | --- |
| Reload | Uncapped |
| Recompute | Cap zero |
| Fixed hybrids | Caps at aligned 25%, 50%, and 75% of the reusable prefix |
| Retrospective selector | Best measured arm per condition; diagnostic only |

The retrospective selector is an optimistic headroom estimate across repeated
arms, not a realizable per-request oracle and not an implemented controller.
Select the best static arm using a calibration split, then assess condition
selection on separate repetitions. Never choose and evaluate the winner on
the same noisy samples.

Each hybrid loads a prefix and recomputes its remaining suffix. This is not
CacheFlow's layerwise restoration algorithm, and it does not cancel or revise
transfers already in flight. A negative result rules out useful switching only
within this tested family and these conditions.

Before each measured arm, reconstruct the same cache state: reset all cache
tiers, prefill the target prefix, wait for stores to complete, and clear the
GPU prefix cache while preserving the connector cache. Verify that this last
operation really preserves the CPU tier on the installed revision. If it
does not, use a controlled eviction sequence and verify residency via counters.
Reject a sample with an unexpected GPU hit or missing external-cache hit.

Use identical prompt token IDs, append length, generation parameters, and
background traffic for all arms. Alternate arm order with a fixed seed.
Warm up every prompt length outside the timed interval. Cache preparation,
allocation, and background-process startup are also outside that interval.

## Conditions

Start with 4k, 8k, and 16k reusable prefixes and a 64-token append. The first
stage measures CPU-to-GPU restoration under no background copies and under
bounded H2D copy bursts. Sweep both burst size and spacing: average bandwidth
alone does not capture head-of-line blocking. Record actual background bytes
and timing; identical offered traffic is not identical achieved bandwidth.

Use real TraceLab prompt lengths for a later sampled distribution. Random
token IDs with those lengths preserve shape, not agent semantics; label such
runs accordingly. The existing local trace contains lengths and timings,
not a faithful replay of original model inputs or outputs.

Initially generate one token to isolate resumption. Repeat promising conditions
with multi-token generation to measure interference with decode. A separate
arrival-rate experiment is needed to assess serving throughput; a batch-one
latency test cannot establish unchanged throughput.

## Correctness and accounting

- Verify external loaded tokens and locally computed tokens for every arm,
  including cap alignment; cap zero must perform no external loading.
- Verify GPU-hit tokens do not differ across arms.
- Compare greedy output token IDs across restoration arms. Investigate any
  difference rather than treating a faster incorrect path as a gain.
- Measure request arrival to first token, completion latency, transfer bytes,
  cache-hit counters, and background traffic. Preserve individual samples.
- Check that the contention process is stopped in `finally`, with a bounded
  join and termination fallback; do not leave GPU work running after errors.

## Decision gate

Calibration chooses the strongest static cap. Separate evaluation repetitions
test whether choosing caps by contention condition improves on that static
baseline, as well as on uncapped reload. Use enough repetitions to report
uncertainty; a handful of samples cannot support a p95 claim.

Proceed toward a controller only if there is repeatable switching between
preferred arms and at least 15% lower p95 resumption latency versus the best
static arm in a prespecified mixture of conditions. This is a target, not an
expected result. Then validate throughput, fairness, and generation correctness
in a serving workload before claiming a system improvement.

Stop or narrow scope if one static arm wins everywhere, or if apparent gains
come from unmatched cache state, changed background traffic, or measurement
noise. A large gain against a deliberately overloaded reload baseline alone
does not meet the gate.

## Second stage: actual staged storage path

CPU-to-GPU contention is only the first component. The proposed direction
also requires secondary-to-CPU-to-GPU restoration, limited staging buffers,
and competing writes. Modal container storage or a mounted Modal Volume must
not be described as local NVMe without verifying the backing hardware and
cache behavior. Use measured storage costs and clearly identify warm page
cache versus direct/cold reads. If appropriate hardware is unavailable, report
the first-stage result as H2D-only and leave the staged-path claim unvalidated.

## Status

Modal SDK installed in a temporary uv environment. Short-lived workspace
authentication verified. The user authorized a $1 initial budget.
`modal_restoration.py` launches `kvplace/restoration_gpu.py` with a published
vLLM 0.30.0 wheel plus upstream commit
`75dc5882699be65eaee144cb7d0b06c1f86d3cec`'s load-cap backport. It uses a
480-second function timeout (450-second subprocess timeout), native sampling,
and eager execution. The first GPU run completed 60 samples with cache and
output checks passing; a repeat warms each load fraction first. This initial
probe uses three repetitions per arm, synthetic token IDs, and H2D-only
contention. It cannot establish the p95 gate or throughput preservation.

The warmed repeat is complete: see [RESTORATION_RESULTS.md](RESTORATION_RESULTS.md).
Full reload wins in all four tested conditions; all 60 warmed samples pass
cache-accounting, output, and background-overlap checks. The tested action
family offers no switching benefit on this GPU/model pair. No further GPU
runs are scheduled.

Reproduce from the repository root:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python modal==1.6.1
.venv/bin/modal token new
.venv/bin/modal run research/modal_restoration.py --out /tmp/restoration.json
```

Running this command incurs Modal charges; use the bounded configuration
and inspect workspace usage. The compatibility patch is saved in
`scripts/native_load_cap.patch`; image building fails if it cannot apply
cleanly. No full source or CUDA compilation is required. The first successful
run observed a first-use transfer-kernel JIT spike in its first 75%-load sample;
the repeat explicitly warms every length/cap combination before timing.

AI assistance: Codex prepared this experiment design. A human must review the
experiment and any serving changes before an upstream submission.
