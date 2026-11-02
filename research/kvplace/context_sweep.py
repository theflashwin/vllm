"""Serving capacity under context limits: max sessions/s that meet a TTFT SLO.

Replays each trace at a grid of arrival rates under each context policy:
    none              the trace as recorded
    fixed:T           sessions over T tokens always drop to keep * T
    pressure:T:D      the same, only while the oldest queued request has waited
                      at least D seconds

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

from kvplace.sim import ContextPolicy, CostModel, Simulator, _pct, get_policy
from kvplace.trace import read_trace


def parse_policy(name: str, keep: float) -> ContextPolicy | None:
    if name == "none":
        return None
    kind, limit, *rest = name.split(":")
    wait = float(rest[0]) if kind == "pressure" else None
    return ContextPolicy(int(limit), keep, wait)


def run_one(job: dict) -> dict:
    sessions = read_trace(job["trace"])
    span = max(s.start_s for s in sessions)
    # Same 300 sessions, packed into the arrival span that gives this rate.
    factor = (len(sessions) / job["rate"]) / span
    for s in sessions:
        s.start_s *= factor
    original = sum(sum(s.prompt_lengths()) for s in sessions)
    n_sessions = len(sessions)
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
        context_policy=parse_policy(job["policy"], job["keep"]),
    )
    result = sim.run()
    ttft = [r.ttft_s for r in result.records]
    by_session: dict[str, list] = {}
    for r in result.records:
        by_session.setdefault(r.session_id, []).append(r.arrival_s)
    dropped_sessions = sum(1 for s in sessions if s.meta.get("context_epoch"))
    return {
        **{k: job[k] for k in ("trace", "policy", "rate", "gpu_blocks", "keep")},
        "p50": _pct(ttft, 50),
        "p90": _pct(ttft, 90),
        "p99": _pct(ttft, 99),
        "session_s": statistics.fmean(max(v) - min(v) for v in by_session.values()),
        "drops": result.link.get("context_drops", 0),
        "sessions_dropped": dropped_sessions / n_sessions,
        "context_lost": 1 - sum(r.prompt_tokens for r in result.records) / original,
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
    p.add_argument("--cost-json", required=True)
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
