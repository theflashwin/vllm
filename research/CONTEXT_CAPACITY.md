# Serving agents under overload: context limits and turn-aware scheduling

Status: simulator results, plus one small live check of per-chain scheduling (Nov 2026).
Context-limit results have no live replay and no quality measurement; "context lost" is
a proxy, not task quality.

Findings so far:

- Truncating context raises capacity, at a high cost in context lost. It is not new.
- Tool-aware alternatives add little: re-fetchable dropping about 0%, oracle
  memory-time queueing +3-22%.
- Scheduling by human-delimited chain needs no prediction or context loss. It gives
  human turns up to about 20x lower p90 TTFT under heavy simulated overload and 3x in
  the live check. Long chains pay for this.

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

## Extra work after a drop (`results/ctx/reread5.json`, `reread20.json`)

Each drop adds N turns of 2000 new tokens and 100 output tokens. These are assumed
values, not measurements. Capacity gain at p90 TTFT <= 5 s, with baselines from the
same grid (TraceLab samples 1 and 3, then CATraces; `>=` means the rate grid's ceiling):

| Policy | 0 turns | 5 turns | 20 turns |
|---|---:|---:|---:|
| fixed 64k | +149, +126, +164% | +144, +109, +83% | +102, +68, +16% |
| pressure 64k, 2 s | +102, +95, +101% | +111, +68, +59% | +87, +39, +13% |
| fixed 32k | +512, +439, >=+166% | +382, +258, >=+99% | +200, +126, +47% |

The 0-turn column comes from a finer rate grid, which shifts baselines by up to 16%,
so small differences from the 5-turn column are not meaningful. On TraceLab, 5 turns
per drop is 9-10 extra turns per session at fixed 64k. CliffCompaction reports +4.68
steps per instance, on much shorter tasks. CATraces is more sensitive than TraceLab.

## Tool-aware alternatives (`results/ctx/select_*.json`, `memtime.json`)

Same setup, limit 64k, p90 TTFT <= 5 s, TraceLab samples 1 and 3, then CATraces.
`rolling` clears every tool result older than 12 turns, in the style of AKV. `refetch`
clears only Read, Grep, Glob, LS, WebFetch, WebSearch and NotebookRead results. "Cache
kept" assumes every cached block survives the edit, an upper bound.

| What is dropped | Gain | Cache kept | Lost |
|---|---:|---:|---:|
| Truncate to 32k | +149, +126, +164% | +183, +156, +247% | 52, 52, 48% |
| Rolling 12 | +2, 0, +15% | +24, +14, +35% | 22, 21, 34% |
| Re-fetchable only | -1, -5, 0% | +8, +1, +27% | 8, 5, 13% |

- Both selective modes miss the 15% bar on TraceLab. Re-fetchable results are only
  11-20% of token-turns; the first turn is about a third, model output 10-27%.
- `refetch_tail` matched `refetch` exactly, so the excess was never met and drop
  placement was not actually tested.
- Oracle shortest-memory-time queueing: +11, +3, +22%, with p99 up to 3x worse on
  TraceLab sample 1 at 0.008 sessions/s.
- Fixed `keep_recent` 12 and `min_clear` 4000; other values untested. Under p90 <= 2 s,
  -100% means the lowest rate tested already missed the SLO.

## Turn-aware queueing and tool-wait timing (`results/ctx/turnclass_*.json`, `gated.json`)

Human-first queueing admits turns that follow a human message or start a session
(6-10% of turns) ahead of agent turns, without preemption. No limit on context.

| Trace, sessions/s | Human p90, FIFO to human-first | Agent p90, FIFO to human-first |
|---|---:|---:|
| TraceLab s1, 0.008 | 142 to 16.6 s | 155 to 184 s |
| TraceLab s1, 0.015 | 463 to 29.6 s | 491 to 516 s |
| CATraces, 0.02 | 194 to 26.1 s | 199 to 195 s |
| CATraces, 0.03 | 296 to 57.4 s | 297 to 306 s |

Mean session time stays within 3%. This does not raise capacity: at human p90 <= 5 s the
gain is +5-7%, and agent queues still grow. It shields human turns under overload.

Truncating only after a tool wait of 10 s or 60 s, with a hard cap, halves drops but is
dominated: 42-46% lost for +75-101% on TraceLab, where pressure 64k, 2 s loses 32% for
+95-102%.

## Human-first with a pressure limit (`results/ctx/combo_*.json`)

Capacity gain against FIFO with no limit, requiring human p90 <= 10 s and agent p90 <=
60 s (TraceLab s1, s3, then CATraces). Lost is measured at capacity.

| Limit | Queue | Gain | Lost |
|---|---|---:|---:|
| None | Human-first | +19, +21, +7% | 0 |
| Pressure 64k, 2 s | FIFO | +125, +117, +100% | 37, 33, 32% |
| Pressure 64k, 2 s | Human-first | +200, +177, +179% | 40, 38, 36% |
| Pressure 64k, 2 s | Human-first, preemption | +198, +176, +177% | 40, 38, 36% |

- FIFO is held back by the human SLO, so part of this gain comes from allowing agent
  turns a looser SLO. At TraceLab s1, 0.02 sessions/s, with the limit, human p90 goes
  from 119.7 s to 7.3 s and agent p90 from 131 s to 142 s.
- Preemption matters only at human p90 <= 5 s: +224, +217% against +62, +117% on
  TraceLab with agent p90 <= 60 s. CATraces is unstable at that SLO (+13% vs +187%)
  because human p90 sits near 5 s across rates.
- Exempting human turns from drops protects little: human-turn context lost is 31, 33,
  33% against 31, 34, 38%, since earlier agent-turn drops persist.
- Preemption resumes reload from the tiers; vLLM's real preemption cost is unmeasured.

## Human-delimited chain scheduling (`results/ctx/chain2_*.json`)

A chain runs from a human message to the agent's final reply. Fewest-turns-first admits
the request whose chain has had the fewest turns; the per-session version counts turns
since the session began and only approximates per-program schedulers such as Autellix.
No context is dropped and nothing is preempted. Chain times exclude tool time.

| Trace, sessions/s | Queue | Human turn p90 TTFT | Short chain p90 | Long chain p90 | Mean session |
|---|---|---:|---:|---:|---:|
| TraceLab s1, 0.0125 | FIFO | 438 s | 1205 s | 15110 s | 13023 s |
| | Per session | 152 s | 538 s | 14850 s | 11778 s |
| | Per chain | 19 s | 269 s | 19558 s | 11800 s |
| TraceLab s3, 0.02 | FIFO | 510 s | 1427 s | 14340 s | 20634 s |
| | Per session | 314 s | 787 s | 14965 s | 19243 s |
| | Per chain | 21 s | 263 s | 19285 s | 19794 s |
| CATraces, 0.03 | FIFO | 296 s | 959 s | 6642 s | 31854 s |
| | Per session | 154 s | 304 s | 3861 s | 26293 s |
| | Per chain | 28 s | 114 s | 1396 s | 22934 s |

- Below capacity the orders match. Short means at most 5 turns.
- Long chains pay on TraceLab: per chain is 18-91% worse at p90 than per session at
  0.0125-0.02 sessions/s, and overall chain p90 can exceed FIFO at high load. No aging
  is modeled, so starvation is unbounded.
- Tokens reloaded from CPU or disk fall 13-19% against FIFO in these three rows, which
  fits less cache churn; causation is not isolated.
- Admitting chains in progress first starves human turns (up to 9032 s p90).

### Aging, all five TraceLab samples (`results/ctx/aging_*.json`)

Mean of 5 samples at 0.0125 sessions/s. With aging, every 60 s of waiting counts as
one turn fewer. Human p90 under per-chain is 4-6% of FIFO's in every sample at this
rate and above, and 9-31% at 0.008.

| Metric | FIFO | Per chain | Per chain, aging 60 s |
|---|---:|---:|---:|
| Human turn p90 TTFT | 365 s | 18 s | 18 s |
| Short chain p90 | 1081 s | 261 s | 262 s |
| All chains, median | 412 s | 131 s | 131 s |
| All chains, mean | 1896 s | 1586 s | 1586 s |
| Long chain p90 | 11486 s | 15567 s | 13873 s |
| Long chain p99 | 24468 s | 31839 s | 31443 s |
| Longest single TTFT | 531 s | 27100 s | 3567 s |
| Mean session | 14614 s | 13592 s | 13598 s |

Aging 60 s bounds starvation far better than none, but the longest wait is still 7x
FIFO's. At 0.03 sessions/s, aging 60 s lets human p90 rise to 58 s from 22 s.

### Live check on vLLM's priority scheduler (`results/live/chain50_a10g/`)

A10G, Qwen2.5-7B, vLLM 0.31, first 50 sessions of `live70` (32k context cap) with
arrivals compressed, 3800 GPU blocks, 14.7 GB CPU. FIFO is `fcfs`; per chain is
`--scheduling-policy priority`, with priority set to the turns since the last human
message. One run per arm, no aging, $1.35. Simulator values were recorded before the
run.

| Metric | Simulator FIFO to per chain | Live FIFO to per chain |
|---|---:|---:|
| Human turn p50 TTFT | 6.1 to 1.4 s | 5.0 to 1.8 s |
| Human turn p90 TTFT | 19.1 to 9.8 s | 19.5 to 6.8 s |
| Short chain p90 | 89 to 92 s | 86 to 79 s |
| Long chain p90 | 506 to 616 s | 521 to 631 s |
| All turns p90 TTFT | 24.3 to 31.8 s | 21.5 to 37.8 s |

- Human TTFT ratio with 95% session bootstrap: p50 0.36 (0.27-0.47), p90 0.35
  (0.24-0.72). Faster in 95 of 120 paired human turns. This excludes run-to-run noise.
- Agent turn p90 is 1.86x FIFO's (1.54-1.96), and the longest TTFT rises from 39 s to
  171 s. Mean chain time rises from 134 s to 157 s, unlike heavier simulated overload.
- Tokens hit in GPU rise 19% (5.59M to 6.66M) and those from CPU or disk fall 13% (8.45M
  to 7.38M), in line with the simulator's 13-19% drop in reloads.
- Overload here is mild, so the 20x simulated at 0.0125 sessions/s is not tested. Per
  session, aging and full-length contexts were not run live. Preemption count not
  captured.

## Not modeled, all of which favor the policy

- Task quality, and whether the assumed extra work matches real agents.
- An LLM call for summarizing: the model assumes dropping tool results, no call.
- Client cooperation and retry time.

Also untested: other keep fractions and smaller GPU memory (sweeps cancelled).

## Next

1. Live per-session baseline, to test the human boundary directly.
2. Heavier live overload and full-length contexts.
3. Aging in a router, since vLLM's request priority is fixed on arrival.
