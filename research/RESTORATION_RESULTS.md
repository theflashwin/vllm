# GPU restoration probe: no crossover in the tested H2D regime

## Finding

Full reload is the fastest of the five measured restoration options in all
four conditions. Selecting a different option by contention condition offers
no improvement over always reloading in this experiment. This is evidence
against building a switching controller for this specific model/GPU/H2D
regime, not a rejection of all restoration scheduling.

## Warmed repeat

One NVIDIA L4, Qwen2.5-1.5B-Instruct BF16, TP1, batch one, eager execution,
native sampling, 1 GiB CPU cache, 16-token blocks, and 1,024 GPU blocks.
Each request resumes a 4,096- or 8,192-token prefix with a 64-token append,
then generates one token. Three randomized-order repetitions per condition
and option produce 60 samples. Every length/load-cap combination is warmed
before timing.

| Prefix tokens | Competing H2D copies | Reload ms | 25% loaded ms | 50% loaded ms | 75% loaded ms | Recompute ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | No | 54.89 | 214.40 | 158.00 | 93.23 | 261.49 |
| 4,096 | Yes | 59.81 | 219.54 | 163.71 | 99.20 | 271.88 |
| 8,192 | No | 74.94 | 481.64 | 342.22 | 203.78 | 589.78 |
| 8,192 | Yes | 84.33 | 497.83 | 359.86 | 212.23 | 619.64 |

Values are median wall time from the start of `LLM.generate` to its return
with one output token, including client/engine round-trip overhead. Engine
first-token latency is recorded separately in the JSON. These are not p95
estimates or task-completion results.

The competing process offers 32 GiB in 128 MiB copies, eight copies per burst,
32 bursts, on a separate CUDA stream/context. The same bytes are offered in
every contended arm; achieved duration is recorded separately. Its active
interval spans each measured request in the warmed run. This is controlled
copy contention, not concurrent model-serving demand or a calibrated queue
of production KV transfers.

The best mixed option (75% loaded) is still about 66% slower than reload at
4k with contention, and 152% slower at 8k with contention. Pure recomputation
is about 4.5x and 7.3x as slow, respectively. Contention increases reload's
median by approximately 9% at 4k and 13% at 8k, which is insufficient to make
any tested mixed/recompute arm preferable.

## Correctness and controls

All 60 warmed samples passed:

- GPU-cache hits equal zero after the GPU-only reset.
- External loaded tokens equal the requested aligned cap, or the whole
  reusable prefix for uncapped reload.
- Locally computed tokens equal prompt length minus externally loaded tokens.
- Greedy output token IDs match across all arms for each prompt length.
- Background bytes equal 32 GiB in every contended sample and zero otherwise.
- Recorded background overlap equals the whole measured request interval.

This output check covers one generated token on two synthetic prompts;
it is not a model-quality evaluation on real agent tasks.

## Artifacts and provenance

- [restoration_results.json](restoration_results.json): authoritative warmed
  run, metadata, 60 individual samples, and medians.
- [restoration_results_initial.json](restoration_results_initial.json): earlier
  exploratory run. Its first 75%-load sample included a transfer-kernel JIT
  spike; retain for audit, use the warmed repeat for the table above.
- [RESTORATION_PLAN.md](RESTORATION_PLAN.md): scope, gate, and reproduction.
- [modal_restoration.py](modal_restoration.py): one-shot Modal launcher.
- [kvplace/restoration_gpu.py](kvplace/restoration_gpu.py): GPU harness.

Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
vLLM 0.30.0, PyTorch 2.13.0+cu130, CUDA 13.0. The published vLLM wheel lacks
the load cap, so the experiment image applies the exact upstream Python patch
from `75dc5882699be65eaee144cb7d0b06c1f86d3cec`, saved in
`scripts/native_load_cap.patch`. No local engine source or CUDA kernel is
changed. The default FlashInfer sampler required unavailable `nvcc`; all
successful measurements instead use native sampling.

Warm-run app: `ap-fDzgBi9f0WvW5707HUl2Mi`; initial successful app:
`ap-15s9294vpoBGMOSxJcyndt`. Three earlier setup attempts also stopped.
All five experiment apps were verified stopped after the repeat. The posted
per-app total was $0.17842979 when checked; billing can lag completion.
The user-authorized initial limit was $1. No more GPU runs are scheduled.

Warmed harness SHA-256:
`db3905b700c3b587ed9608ef00549ce82ed9bd20da644f75f386646ba3496987`.
Launcher SHA-256:
`7df7c5080dbd39fb768d3ae7f924a12aa6dd1b2b008db2f7b1aa4b34e75a0a8f`.

## Limits and next decision

This tests prefix-load/suffix-recompute caps chosen before a request starts.
It does not test layerwise recomputation, changing decisions mid-transfer,
transfer cancellation, staging-buffer pressure, secondary-storage reads,
competing writes, scheduling fairness, throughput, or tail SLOs. The model
uses GQA; its compute-to-KV-byte ratio and this L4's performance need not
represent other architectures or faster GPUs. Inputs are seeded random token
IDs, not a replay of TraceLab's original contents.

Do not implement a controller for the tested regime. A further experiment
would need actual staged-storage contention or a materially different
compute/transfer ratio, with matched cache state and measured costs. No
latency gain or novel scheduling mechanism has been established here.

## Local validation

`PYTHONPATH=research .venv/bin/python -m pytest research/tests/test_sim.py
research/tests/test_tracelab.py -q -p no:cacheprovider`: 12 passed (the local
temporary uv environment was used).

Ruff check/format on the new harness and launcher, and `git diff --check`,
passed. GPU validation ran through the launcher's temporary uv environment
inside Modal. AI assistance: Codex designed, implemented, ran, and documented
this exploratory experiment; human review is required before upstream work.
