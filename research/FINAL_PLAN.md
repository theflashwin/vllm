# Final plan: eviction-only paper (written 2026-11-09)

Status: plan. Scope is CPU-to-disk eviction only. Retention economics, context
limits, chain scheduling and multi-GPU routing are out of the paper.

Dates below assume deliverables are due the week of Dec 7. **Confirm the date.**
If it is Nov 30, drop E4 and the 1.1 load level and move everything up a week.

## Thesis

> Agent-aware eviction below the GPU saves disk I/O and recompute, not latency,
> because GPU admission sets TTFT.

Midpoint hypothesis: evicting the sessions predicted to return last cuts p90
resume TTFT by at least 10% when KV overflows CPU memory. Status: disproved at
sustainable load. The talk and paper open with that hypothesis and close on why
it failed.

## Claims and evidence

Every claim rests on live runs. The simulator is a mechanism: it searches for
settings and predicts the sign of each live run. It is not a finding.

| | Claim | Live evidence we have | Still needed |
|---|---|---|---|
| C1 | At sustainable load there is little for any CPU-tier eviction policy to gain | None yet | E2 (bound) |
| C2 | Predicted-return eviction does not lower TTFT against LRU, because requests wait for GPU memory | A100, both run orders: same TTFT. A10G stable, one run per arm: ours p90 1.23 s vs LRU 0.79 s | E3 (repeats, TTFT split into queue, load, prefill) |
| C3 | Under disk pressure it does save I/O and recompute | A100, both run orders: 35% fewer disk reads, 23-28% fewer recomputed tokens | Raw runs are not in `results/live`; recover them or rerun. Same counters from E3 |

Not claims:

- **The simulator and its admission model.** They go in the methodology
  section, with one table validating the simulator against live runs. That the
  midpoint numbers came from a simulator without admission belongs in the
  timeline and lessons-learned part of the talk.
- **70B and shared slow disks.** Simulator only (+8-10% at best), so they are
  discussion, marked simulated, unless E4 confirms one live.

Things to state plainly in the paper:

- Live baselines are stock vLLM (LRU on GPU and CPU). The "agent-aware GPU
  only" baseline from the midpoint exists only in the simulator.
- Live runs so far use token scale 0.15, a 32k context cap and time scale
  0.2-0.5. E0 decides how much of that we can remove.
- Model: PACE-ICE prohibits Qwen, so every run from E0 on uses Llama 3.1 8B
  Instruct on an A100, with the CPU tier scaled to hold the same number of
  tokens (4.19 GB; 128 KiB of KV per token against Qwen2.5-7B's 56 KiB). The
  earlier Qwen runs (A10G, A100) are context, not headline evidence. The
  simulator's cost files were measured for Qwen: remeasure them in E0.

## Experiments

About 35-40 one-hour arms in total. Run them on PACE-ICE; keep Modal (under $30)
for the demo and one cross-check.

**E0. PACE-ICE bring-up (Nov 9-13).** Schedule risk, start first.
- Launcher: `scripts/pace_live.sbatch` wraps `scripts/run_baselines.sh`. It
  refuses a RAM-backed or buffered disk tier, checks the CPU tier fits the
  job's memory, and writes `node.txt` and `args.json` next to each run. Not
  yet run on a cluster: start with `DRY_RUN=1` on a GPU node and confirm the
  partition and GPU names with `sinfo -o "%P %G %l %m"`.
- Measure costs on the node: `kvplace.microbench gpu`, `fs --dir <FS_DIR>`,
  `server`. Check that the FS tier's storage supports O_DIRECT.
- Raise the context cap as far as the card allows (`decode_scan.py` served a
  160k request on the A100) and record the token scale used.
- Hint overhead (midpoint feedback): run `tier_lru` once with `HINT_PORT=`
  (hint-only requests, as before) next to the default (UDP hints). The
  difference in resume TTFT is what the hint-only requests cost the engine.
  See "Midpoint feedback".
- Exit: two `tier_lru` runs of `live35` agree within 10% at p50 and p90, and
  the `tier_reuse` summary shows the policy read about as many UDP hints as
  the client sent (the last few can be unread when the run ends).

**E1. Sustainable load (Nov 13-15).** `tier_lru` at 3-4 arrival rates
(`tracelab --rebase-rate`). λ\* is the highest rate with p90 resume TTFT <= 5 s.
All later runs are at 0.5, 0.85 and 1.1 λ\*.

**E2. Opportunity bound (Nov 16-19).** No new policy code. Same hint traffic in
every arm (`PREDICTOR=tool-nfh2`).

| Arm | Config | Meaning |
|---|---|---|
| Recompute | `gpu_only` | Every GPU miss is recomputed |
| Small CPU tier | `tier_lru`, CPU tier far below idle KV | What LRU gets today |
| All in DRAM | `cpu_lru`, CPU tier >= all unique KV | Best any CPU-tier eviction policy could do |

At 0.5 and 0.85 λ\*, two repeats. The gap between the last two arms is the most
any eviction policy can gain.
- Gap under 10% at p90: C2 holds for every policy, not just ours.
- Gap of 10% or more: headroom exists. E3 then shows whether prediction
  captures it, and the thesis changes to match.

**E3. Policy comparison (Nov 16-22).** `tier_lru` vs `tier_reuse`, three loads,
three repeats, alternating which arm runs first. Report with a 95% session
bootstrap (as in the chain analysis):
- resume TTFT p50, p90, p99, and the ratio ours / LRU;
- TTFT split from `vllm:request_queue_time_seconds`, `request_prefill_time`
  and `request_decode_time` (the uncommitted `replay.py` change collects these);
- bytes read from disk, recomputed tokens, share of resumes that load from disk.

This settles whether the +56% loss at stable load is real. Write down the
expected sign from the simulator before each run, as was done for chain50.

**E4. Slow-disk boundary (Nov 23-25, optional).** Only if E0-E3 are done by
Nov 22. Rerun the E3 pair at 0.85 λ\* with the FS tier on the slowest storage
that supports O_DIRECT. Confirms or rejects the simulated +10%.

**E5. Simulator support (parallel, $0).** Supports the live runs; produces no
headline numbers.
- Predict each E2 and E3 arm before it runs, and pick the CPU tier sizes and
  rates worth running live.
- Validation table for the methodology section: simulator against live on the
  same settings, including the new E3 runs.
- Rerun the 70B and shared-disk cells on five samples for the discussion.
- Every result file records the commit it came from.

## Figures and tables

Every latency result is an empirical CDF of resume TTFT (log x-axis, with a
tail panel above p90), pooled over repeats, with p50, p90 and p99 marked. No
p90-only bar charts (midpoint feedback).

1. Key result: resume TTFT CDFs, LRU vs ours, at three loads, with the TTFT
   split beside them (E3).
2. Opportunity bound: TTFT CDFs for the three arms at two loads (E2).
3. Share of resumes that load from disk at each load, from the live runs (E3).
4. What eviction does save: disk bytes and recomputed tokens (E3, A100 runs).
5. Methodology: simulator validated against live runs (E5).
6. Discussion: 70B and shared slow disk, marked simulated.
7. Table: what we tried and dropped (prefetch, timed warm-up, bimodal
   predictor, LRU fallback), one line each.

## Midpoint feedback (9.5 / 10)

Two notes cost points. Both get a visible answer in the final.

**Hint-only requests (-0.2).** "Ensure the two-request hint injection
workaround is phased out so it doesn't add engine overhead in live runs."
- What it costs today, from `live35_stable_a10g`: one hint request per turn
  (811 of 1,622 requests), median 0.17 s each, p99 8-13 s. Each one takes a
  scheduler slot and stores one padding block.
- Why it exists: the tool is known only after the response, and in tiered
  mode the CPU tier only sees requests that store or load a chunk.
- Both arms of every comparison receive the same hint traffic, so ratios are
  fair. Absolute TTFT is not.
- Replacement (written Nov 10, not yet run on a real server): the client
  sends the estimate as a UDP datagram to the policy (`KVPLACE_HINT_ADDR`,
  `replay --hint-addr`), and the policy reads pending datagrams before each
  eviction and when a request finishes. No request enters the engine and no
  padding block is stored. `pace_live.sbatch` uses it by default;
  `modal_live.py` takes `--hint-port`.
- Every run from E0 on uses UDP hints. Earlier runs used hint-only requests,
  so their absolute TTFT is not comparable; `args.json` records
  `hint_transport`.
- Limitation to state in the paper: client and server must share a host. A
  deployment would need a hint endpoint on the server.

**CDFs (-0.3).** "Supplement the scalar p90 bar charts with full empirical
CDFs." See the rule at the top of "Figures and tables". `requests.csv`
already has per-request TTFT, so this is plotting only.

The reviewer also praised the comparison across hardware configurations.
Those numbers came from the simulator without GPU admission and do not hold.
The timeline slide has to say so directly.

## Schedule

| Week | Live track | Simulator and figures | Paper and artifact |
|---|---|---|---|
| Nov 9-15 | E0, E1 | Commit the index fix; predictions for E2 and E3 | LaTeX skeleton with the three claims; recover the A100 runs from the `kvplace-results` Modal volume |
| Nov 16-22 | E2, E3 | Validation table; five-sample 70B and shared-disk reruns | Background, related work, design and implementation sections |
| Nov 23-29 (Thanksgiving) | E4 if on track; otherwise finish E3. **Freeze experiments Nov 29** | Final figures | Evaluation section; README and examples |
| Nov 30-Dec 6 | Reruns for broken arms only | Demo script | Slides, two timed rehearsals, contribution statement, push to the course repo |

## Repo and artifact work

- Commit the working-tree `vllm_policy.py` and `replay.py`. HEAD's `evict`
  scores every chunk on each call, which stalled the scheduler in the live run;
  the working tree matches `origin/datta/next`.
- Rewrite `README.md`: setup, tests, one command per figure. It still describes
  weeks 1-2 and points at `DIRECTION.md`.
- Move notes for dropped directions into `research/notes/` so the top level
  shows only what the paper uses.
- Add `research/examples/`: the policy loaded in vLLM's CPU offload manager;
  one live replay of LRU against ours; a one-minute simulator run.
- `reproduce.sh` (or a Makefile) that regenerates each figure from committed
  results, and says which runs need a GPU.
- Paper repo that builds with `make`. Copy of the code in the course GitHub org.

## Paper outline (MLSys format)

1. Introduction: agents pause constantly; idle KV overflows memory; hypothesis.
2. Background and related work: tiered KV offload in vLLM; CacheWise,
   TokenCake, Continuum, MORI; the eviction studies that agree with us
   (Fancy Eviction, Where Should the KV Cache Live, AgentServeSim).
3. Design: return-time prediction, the hint path, eviction rule.
4. Implementation: hint-only requests and the padding block; the scheduler
   stall and the per-owner index; assumptions (write-through disk tier,
   per-tool medians) and the limitations that follow.
5. Methodology: traces, live replay harness, load relative to λ\*; the
   simulator as the search tool, its admission model, and its validation.
6. Evaluation: figures 1-4, all live.
7. Discussion: why admission dominates; where to look next (70B, shared slow
   storage, simulated).
8. Conclusion.

Talk (14 min): problem 2.5, state of the art 1.5, design 2, implementation 2,
evaluation 4, timeline 1, demo and summary 1. The simulator appears twice: as
a tool in implementation, and in the timeline as the reason the midpoint
numbers changed. The demo is the policy running in vLLM on a short replay, with
hint logs and the disk-read counters for LRU and ours side by side.

## Risks

- **PACE-ICE port slips.** Fallback by Nov 15: A10G on Modal, E3 at two loads
  with two repeats (about 8 arms, roughly $10), E2 at one load.
- **E2 shows headroom or E3 shows a win.** Report it. The thesis is then scoped
  to the loads where it holds.
- **"So what does matter?"** One discussion sentence can cite the existing
  live chain-scheduling result as contrast. Team decision; no new experiments.
- **Scaled-down traces.** State the token and time scales next to every live
  number. Full-scale simulator results go in the discussion, marked simulated.

## Open decisions

1. Exact deadline and presentation date.
2. Venue format: MLSys (recommended) or OSDI.
3. Whether the discussion mentions chain scheduling.
4. Who owns each track in the schedule.
