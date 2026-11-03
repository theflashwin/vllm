"""Serving capacity under context limits: max sessions/s that meet a TTFT SLO.

Replays each trace at a grid of arrival rates under each context policy:
    none              the trace as recorded
    fixed:T           sessions over T tokens always drop to keep * T
    pressure:T:D      the same, only while the oldest queued request has waited
                      at least D seconds
    fixed:T:S         S says what to drop instead: rolling, refetch or refetch_tail;
                      a _free suffix (truncate_free too) assumes the cache survives
    pressure:T:D:S    the same, under pressure
    pressure:T:D:agent truncate, but never at turns that follow a human message
    fixed:T:gated:W:C truncate only after a tool wait of W seconds, or above C tokens

Example:
    python -m kvplace.context_sweep traces/fx/tl300_sp_s1.jsonl \
        --policies none fixed:64000 pressure:64000:2 --rates 0.05 0.1 0.2 \
        --gpu-blocks 66500 --cost-json costs_scan_a100.json --out sweep.json

Block counts are in vLLM's 16-token blocks; `--block-tokens` simulates coarser
blocks of the same total memory, which is much faster at long contexts.

"""

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from kvplace.sim import (
    REREAD_TOOL,
    ContextPolicy,
    CostModel,
    Simulator,
    _pct,
    get_policy,
    is_human_turn,
)
from kvplace.trace import read_trace

SHORT_CHAIN_TURNS = 5


def parse_policy(
    name: str, keep: float, reread_turns: int, reread_frac: float = 0.0
) -> ContextPolicy | None:
    if name == "none":
        return None
    kind, limit, *rest = name.split(":")
    wait = float(rest.pop(0)) if kind == "pressure" else None
    select = rest[0] if rest else "truncate"
    gate = (float(rest[1]), int(rest[2])) if select == "gated" else (None, 0)
    agent_only = select == "agent"
    select = "truncate" if select in ("gated", "agent") else select
    return ContextPolicy(
        int(limit),
        keep,
        wait,
        reread_turns,
        reread_frac=reread_frac,
        select=select.removesuffix("_free"),
        free_edit=select.endswith("_free"),
        gate_wait_s=gate[0],
        gate_cap=gate[1],
        agent_only=agent_only,
    )


def chain_waits(sessions, records, finish_at) -> list[tuple[int, float]]:
    """(turns, seconds from human message to final answer, less tool time) per chain."""
    arrival = {(r.session_id, r.turn): r.arrival_s for r in records}
    chains = []
    for s in sessions:
        starts = [k for k in range(len(s.turns)) if is_human_turn(s, k)]
        for first, end in zip(starts, [*starts[1:], len(s.turns)]):
            tool_s = sum(t.tool_duration_s for t in s.turns[first : end - 1])
            total = finish_at[s.session_id, end - 1] - arrival[s.session_id, first]
            chains.append((end - first, total - tool_s))
    return chains


def run_one(job: dict) -> dict:
    sessions = read_trace(job["trace"])
    span = max(s.start_s for s in sessions)
    # Same 300 sessions, packed into the arrival span that gives this rate.
    factor = (len(sessions) / job["rate"]) / span
    for s in sessions:
        s.start_s *= factor
    original = sum(sum(s.prompt_lengths()) for s in sessions)
    n_sessions = len(sessions)
    original_human = sum(
        n
        for s in sessions
        for k, n in enumerate(s.prompt_lengths())
        if is_human_turn(s, k)
    )
    cost = CostModel.from_json(job["cost_json"])
    scale = job["block_tokens"] // cost.block_tokens
    sim = Simulator(
        sessions,
        get_policy("lru"),
        "none",
        replace(cost, block_tokens=job["block_tokens"]),
        job["gpu_blocks"] // scale,
        job["cpu_blocks"] // scale,
        job["sec_blocks"] // scale,
        contend_sec=True,
        write_model="shared",
        gpu_admission=True,
        context_policy=parse_policy(
            job["policy"], job["keep"], job["reread_turns"], job["reread_frac"]
        ),
        queue_order=job["queue_order"],
        chain_aging_s=job["chain_aging_s"],
    )
    result = sim.run()
    ttft = [r.ttft_s for r in result.records]
    by_id = {s.session_id: s for s in sessions}
    by_session: dict[str, list] = {}
    for r in result.records:
        by_session.setdefault(r.session_id, []).append(r.arrival_s)
    kept_human = sum(
        r.prompt_tokens
        for r in result.records
        if is_human_turn(by_id[r.session_id], r.turn)
    )
    human = [
        r.ttft_s for r in result.records if is_human_turn(by_id[r.session_id], r.turn)
    ]
    agent = [
        r.ttft_s
        for r in result.records
        if not is_human_turn(by_id[r.session_id], r.turn)
    ]
    chains = chain_waits(sessions, result.records, sim.finish_at)
    short = [w for n, w in chains if n <= SHORT_CHAIN_TURNS]
    long = [w for n, w in chains if n > SHORT_CHAIN_TURNS]
    dropped_sessions = sum(1 for s in sessions if s.meta.get("context_epoch"))
    kept = sum(
        r.prompt_tokens
        for r in result.records
        if by_id[r.session_id].turns[r.turn].tool_name != REREAD_TOOL
    )
    return {
        **{k: job[k] for k in ("trace", "policy", "rate", "gpu_blocks", "keep")},
        "p50": _pct(ttft, 50),
        "p90": _pct(ttft, 90),
        "p99": _pct(ttft, 99),
        "chain_p50": _pct([w for _, w in chains], 50),
        "chain_p90": _pct([w for _, w in chains], 90),
        "short_chain_p50": _pct(short, 50),
        "short_chain_p90": _pct(short, 90),
        "long_chain_p50": _pct(long, 50),
        "long_chain_p90": _pct(long, 90),
        "long_chain_p99": _pct(long, 99),
        "chain_mean": statistics.fmean(w for _, w in chains),
        "max_ttft": max(ttft),
        "reload_tokens": sum(
            r.cpu_hit_blocks + r.sec_hit_blocks for r in result.records
        )
        * job["block_tokens"],
        "miss_tokens": sum(r.miss_blocks for r in result.records) * job["block_tokens"],
        "gpu_hit_tokens": sum(r.gpu_hit_blocks for r in result.records)
        * job["block_tokens"],
        "sim_end_s": result.link.get("sim_end_s", 0),
        "human_p90": _pct(human, 90),
        "agent_p90": _pct(agent, 90),
        "session_s": statistics.fmean(max(v) - min(v) for v in by_session.values()),
        "drops": result.link.get("context_drops", 0),
        "sessions_dropped": dropped_sessions / n_sessions,
        "context_lost": 1 - kept / original,
        "human_context_lost": 1 - kept_human / original_human,
        "preemptions": result.link.get("preemptions", 0),
        "refetch_share": result.link.get("dropped_refetch_tokens", 0)
        / max(1, result.link.get("context_dropped_tokens", 0)),
        "extra_turns": sum(
            t.tool_name == REREAD_TOOL for s in sessions for t in s.turns
        ),
    }


def capacity(rows: list[dict], slo: float) -> float:
    """Largest rate meeting the SLO, interpolating p90 in log space."""
    rows = sorted(rows, key=lambda r: r["rate"])
    best = 0.0
    for lo, hi in zip(rows, rows[1:]):
        if lo["p90"] <= slo < hi["p90"]:
            w = (slo - lo["p90"]) / (hi["p90"] - lo["p90"])
            best = lo["rate"] + w * (hi["rate"] - lo["rate"])
    if rows and rows[-1]["p90"] <= slo:
        best = rows[-1]["rate"]
    return best


def _trace_capacity(rows: list[dict], trace: str, policy: str, slo: float) -> float:
    matching = [r for r in rows if r["trace"] == trace and r["policy"] == policy]
    return capacity(matching, slo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("traces", nargs="+")
    p.add_argument("--policies", nargs="+", default=["none"])
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--gpu-blocks", type=int, default=66500)
    p.add_argument("--cpu-blocks", type=int, default=140000)
    p.add_argument("--sec-blocks", type=int, default=10_000_000)
    p.add_argument("--keep", type=float, default=0.5)
    p.add_argument("--block-tokens", type=int, default=256)
    p.add_argument("--reread-turns", type=int, default=0, help="extra turns per drop")
    p.add_argument("--reread-frac", type=float, default=0.0)
    p.add_argument("--cost-json", required=True)
    p.add_argument(
        "--queue-order",
        choices=[
            "fifo",
            "memtime",
            "human",
            "human_preempt",
            "chain_las",
            "chain_sticky",
            "session_las",
        ],
        default="fifo",
    )
    p.add_argument(
        "--chain-aging", type=float, help="waiting this long counts as one turn fewer"
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--slos", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    p.add_argument("--out")
    args = p.parse_args()

    jobs = [
        {
            "trace": t,
            "policy": pol,
            "rate": rate,
            "gpu_blocks": args.gpu_blocks,
            "cpu_blocks": args.cpu_blocks,
            "sec_blocks": args.sec_blocks,
            "keep": args.keep,
            "block_tokens": args.block_tokens,
            "reread_turns": args.reread_turns,
            "reread_frac": args.reread_frac,
            "queue_order": args.queue_order,
            "chain_aging_s": args.chain_aging,
            "cost_json": args.cost_json,
        }
        for t in args.traces
        for pol in args.policies
        for rate in args.rates
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = []
        for row in pool.map(run_one, jobs):
            rows.append(row)
            print(
                f"{row['trace'].split('/')[-1]} {row['policy']} r={row['rate']} "
                f"p50={row['p50']:.2f} p90={row['p90']:.2f} p99={row['p99']:.2f} "
                f"session={row['session_s']:.0f}s drops={row['drops']} "
                f"sessions={row['sessions_dropped']:.2f} "
                f"lost={row['context_lost']:.3f}",
                flush=True,
            )
    print("\ncapacity (sessions/s) at p90 TTFT SLO; gain vs none on the same trace")
    for slo in args.slos:
        print(f"SLO {slo:g} s")
        for pol in args.policies:
            caps, gains = [], []
            for t in args.traces:
                base = _trace_capacity(rows, t, "none", slo)
                caps.append(_trace_capacity(rows, t, pol, slo))
                if base:
                    gains.append(caps[-1] / base - 1)
            lost = statistics.fmean(
                r["context_lost"] for r in rows if r["policy"] == pol
            )
            mean_gain = 100 * statistics.fmean(gains) if gains else 0.0
            print(
                f"  {pol:20s} cap={statistics.fmean(caps):.3f} "
                f"gain={mean_gain:+.0f}% lost(all rates)={lost:.3f}"
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
