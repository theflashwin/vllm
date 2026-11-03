# Quality pilot for context limits: design

Status: design only, nothing run or spent. See `CONTEXT_CAPACITY.md` for the
capacity results this has to justify.

## What published work already answers

[CliffCompaction](https://arxiv.org/abs/2609.26779) (Sep 2026; full paper read) drops
long tool results and reduces tool calls to signatures when context crosses a
threshold, with no LLM call. Resolve rates, full context vs threshold:

| Benchmark, agent, model | Full | 32k | 16k | 8k |
|---|---:|---:|---:|---:|
| SWE-bench Verified, mini-swe-agent, Kimi K2.6 | 73.9% | 73.3% | 71.9% | 67.6% |
| SWE-bench Verified, mini-swe-agent, GLM 5.1 | 71.4% | 68.8% | 69.3% | 65.1% |
| Terminal-Bench 2.0, Terminus-2, Kimi K2.6 | 59.2% | 61.4% | 61.4% | 50.0% |
| Terminal-Bench 2.0, Terminus-2, GLM 5.1 | 49.8% | 53.2% | 54.3% | 44.9% |

[AKV](https://arxiv.org/html/2610.28577v1) (Oct 2026) reports +1.3 to +2.1 points
against standard summarization, with no confidence intervals. Both are preprints.

**A plain replication of these is not worth our money.** Note the consequence for
our claim: if quality really is flat down to 32k, a fixed 32k limit beats any
pressure-driven limit (+481% vs +321% on TraceLab) and adaptivity has no purpose.

## The question left open

Those are single-task, autonomous benchmarks; CliffCompaction explicitly gives up
"full-history recall". Our traces are long sessions with a median context near 80k.
Pressure-driven limits only earn their place if some work really does need old context.

> Is there work where dropped context costs task success or extra turns, and how
> much of the capacity gain survives once that cost is counted?

## Task families

**A. Long autonomous tasks (anchor).** Terminal-Bench tasks, screened with a
full-context run. Keep the 30 that pass 64k peak context, since a limit that never
fires tests nothing. This links us to the published numbers at longer contexts.

**B. Recall-dependent sessions (the new part).** 12 scripted sessions of 3-5 user
turns on real repositories, scored by hidden tests. Each later turn needs something
that cannot be re-fetched: output of a one-off command, a log since deleted, state
that has changed, or a user instruction given mid-session. Family B is built to find
harm, so report it separately as a worst case and never pool it with A.

How common B is cannot be measured from CATraces: `tool_input` is fully redacted
(11,617 of 11,617), so repeat reads are invisible. What the traces do show: tool results
are 56-61% of added tokens, re-fetchable tools are 35-41% of tool tokens, and 113 of 136
CATraces sessions have a later human turn (88-105 of 300 on TraceLab).

Family A risk: SWE-bench and Terminal-Bench trajectories are short, so few tasks may
pass 64k peak. CliffCompaction's KernelBench runs exceed 1M tokens and may be needed.

## Arms

Paired: every task runs in every arm, same model and start state.

1. **Full:** no limit; the harness's own compaction is set above the task's peak.
2. **15% lost:** limit chosen so that replaying Full's context curve loses 15%.
3. **30% lost:** the same for 30%.

Our `context lost` metric maps onto limits differently for each session shape, so
calibrate limits on the Full runs, offline, then check the loss achieved.

Mechanism: follow CliffCompaction's rule (drop results over 500 characters, tool
calls to signatures, keep system prompt, task and recent turns). They publish a
proxy; check its license and that it works with the chosen model. On Claude models
editing history can invalidate thinking blocks, so use server-side context editing
(`clear_tool_uses_20250919`) there, which removes results only. That is a different
mechanism and must be reported as such.

## Measures

- **Primary:** paired difference in task success against Full, by family.
- **Secondary, usually more sensitive:** turns, total input and output tokens, repeated
  tool calls, wall time.
- **Net capacity:** feed the extra work into `context_sweep` as extra turns and report
  capacity gain after that cost. This is the number the project needs. Free once
  the runs exist.

## What a pilot this size can show

With a 25% discordance rate, the paired difference has a standard error of about 9
points at n=30 and 14 points at n=12 (95% interval about +/-18 and +/-28). So this
**cannot confirm a small effect**; it can catch large harm. Published deltas of 1-3
points are far below what we can resolve. Tokens and turns are expected to be more
informative, but their spread is unknown until the screening runs.

## Decision rules (set before running)

| Outcome | Conclusion |
|---|---|
| Family A flat, B clearly harmed | Adaptivity has a case: limit only under pressure, and exempt or warn recall-heavy sessions. Continue to live replay. |
| A and B both flat | Fixed low limits win. Drop the adaptive claim; the result is a validated serving-capacity study of fixed limits. Reassess novelty against CliffCompaction and AKV first. |
| A clearly harmed at 15% | Contradicts published results: audit the setup before believing it. If it stands, the gains shrink to light limits only. |
| Net capacity gain under 15% at any loss with acceptable success | Stop this direction (same bar as the capacity go/no-go). |

"Clearly harmed": success at least 15 points lower with the interval excluding 0, or
at least 25% more tokens per completed task.

## Cost

Runs: 60 screening + 30 x 3 family A + 12 x 3 family B = about 186, one repetition.
Rough per-run assumption: 150 turns at 60k mean context. **API spend is not covered
by the Modal credits.** These are estimates, to be replaced after 5 trial runs.

| Model | Per run (est.) | Pilot (est.) | Note |
|---|---:|---:|---|
| Kimi K2.6 | $0.40-1.20 | $75-225 | $0.40 is the paper's full-context Terminal-Bench cost; screened tasks run longer. Matches published anchor. Provider not chosen. |
| Claude Haiku 5.5 | $0.20-1.00 | $40-190 | Price doubles past 100k prompts. Whether it can solve these tasks is untested: floor-effect risk. |
| Claude Sonnet 5.5 | about $2.40 | about $450 | Likely strongest results; over a class budget. |

Local-only models on Modal (the credits) look like a poor fit: Qwen2.5-7B will likely
hit a floor on these tasks, and larger coding models need GPUs we have not tested.

## Steps

1. **Done:** CliffCompaction read. Its MIT-licensed proxy is public
   (github.com/nguyenvuthientrang/cliffcompaction); no trajectories found.
2. **Done, limited:** CATraces redaction blocks the re-read measure (see above).
3. **Done:** `--reread-turns` in `context_sweep`; results in `CONTEXT_CAPACITY.md`.
4. **About $5:** run 5 tasks end to end, confirm the harness runs here (Terminal-Bench
   needs Docker; not yet tried on this machine), and that limits fire. Fix cost estimates.
5. **Budgeted:** screening, then the paired runs, with a hard API spend cap.
6. Draft family B alongside step 4; have a teammate who did not write a scenario check
   that a full-context agent can actually solve it.
