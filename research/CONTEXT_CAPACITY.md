# Context as a load-driven capacity knob: first results

Status: passes the simulator go/no-go (Nov 2026). No live replay of the policy
and no quality measurement yet. "Context lost" below is a proxy, not task quality.

## Live measurement (`kvplace/decode_scan.py`, `results/scan/`)

Qwen2.5-7B on vLLM 0.31, warmed prompts, random tokens, lockstep decode.

| Quantity | A100 | A10G |
|---|---:|---:|
| Decode step | 10.6 ms + 0.035 us x total context tokens | 32.8 ms + 0.111 us x total |
| Fit | R2 0.99, worst error 4% | R2 0.98, worst error 3% |
| Time per prefilled token | 62 us + 2.62 ns x tokens before it | 214 us + 7.45 ns x tokens before it |
| Cold prefill | 80k: 13.3 s, 160k: 43.5 s | 32k: 10.7 s, 64k: 29.0 s |
| Largest tested | 320k total, 160k single | 64k total |

- Decode depends on total context across requests, as the scan model says. One
  request barely notices its own context: +46% from 2k to 160k on A100.
- Prefill cost is quadratic in context, which the old simulator did not model.
- The A100 run reported a KV pool of 1.06M tokens (about 61 GB). That implies an
  80 GB card although we requested `A100-40GB`. GPU memory size not confirmed.

## Simulator validation (A10G live runs, nothing fitted to them)

| Run | Old p50/p90/p99 | New | Live |
|---|---:|---:|---:|
| `live70`, 14.7 GB CPU | 4.0 / 19.4 / 32.1 | 9.5 / 25.2 / 36.7 | 11.0 / 27.5 / 39.2 |
| `live70`, 3.7 GB CPU | 6.8 / 23.3 / 31.1 | 11.9 / 26.9 / 39.6 | 17.8 / 43.0 / 58.8 |
| `live35`, stable | 0.04 / 0.29 / 3.1 | 0.10 / 0.53 / 4.6 | 0.21 / 0.79 / 9.3 |

The 3.7 GB gap fits promotion thrash, which is still not modeled. Live runs were
capped near 32k context, so full-length contexts are extrapolated.

## Capacity (`kvplace/context_sweep.py`, `results/ctx/a100_k50.json`)

A100 costs, 1.06M token GPU pool, 128 GB CPU, Datta's disk speed. Sessions over T
tokens fall to 0.5 T; only the shared prefix stays cached. `pressure:T:D` applies
this only while the oldest queued request has waited D seconds. Capacity is the
sessions/s meeting p90 TTFT <= 5 s; loss is measured at that load. TraceLab is the
mean of 5 samples. Baseline: 0.0051-0.0062 sessions/s on TraceLab, 0.015 on CATraces.

| Policy | TraceLab gain (range) | Lost | CATraces gain | Lost |
|---|---:|---:|---:|---:|
| fixed 150k | +13% (+3, +18) | 14% | 0% | 6% |
| fixed 100k | +61% (+39, +74) | 32% | +34% | 26% |
| fixed 64k | +151% (+126, +183) | 52% | +164% | 48% |
| fixed 32k | +481% (+427, +529) | 75% | +166% | 73% |
| pressure 100k, 2 s | +43% (+33, +58) | 15% | +33% | 15% |
| pressure 64k, 10 s | +68% (+56, +87) | 23% | +61% | 21% |
| pressure 64k, 2 s | +112% (+95, +138) | 32% | +101% | 32% |
| pressure 32k, 2 s | +321% (+269, +358) | 54% | +166% | 44% |

At equal context lost, pressure-driven limits give roughly twice the capacity
gain of a fixed limit on TraceLab: at 32% lost, +112% vs +61%. Below capacity they
drop nothing, whereas a fixed limit always pays its full loss.

## Not modeled, all of which favor the policy

- Task quality, or extra turns and repeated tool calls after context is lost.
- An LLM call for summarizing: the model assumes dropping tool results, no call.
- Client cooperation and retry time.

Also untested: other keep fractions and smaller GPU memory (sweeps cancelled).

## Next

1. Quality pilot: does task success survive 15-30% context lost?
2. Live replay of none, fixed and pressure at full-length contexts on the A100.
3. Keep fraction and GPU memory sensitivity, on fewer traces and rates.
