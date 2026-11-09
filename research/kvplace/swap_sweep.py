"""Capacity under memory-aware admission and swap preemption.

Example:
    python -m kvplace.swap_sweep traces/fx/tl300_sp_s1.jsonl \
        --arms fifo backfill size swap --rates 0.005 0.006 \
        --cost-json costs_scan_a100.json --out swap.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from kvplace.sim import CostModel, Simulator, SwapPolicy, _pct, get_policy
from kvplace.trace import read_trace

ARMS = {
    "fifo": None,
    "backfill": SwapPolicy(),
    "size": SwapPolicy(order="size"),
    "oracle": SwapPolicy(order="oracle"),
    "swap_fifo": SwapPolicy(preempt=True),
    "swap": SwapPolicy(order="size", preempt=True),
    "swap_oracle": SwapPolicy(order="oracle", preempt=True),
}


def _stats(name: str, xs: list[float]) -> dict:
    return {f"{name}_{p}": _pct(xs, p) for p in (50, 90, 99)}


def run_one(job: dict) -> dict:
    sessions = read_trace(job["trace"])
    span = max(s.start_s for s in sessions)
    factor = (len(sessions) / job["rate"]) / span
    for s in sessions:
        s.start_s *= factor
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
        swap=ARMS[job["arm"]],
    )
    result = sim.run()
    recs = result.records
    ttft = [r.ttft_s for r in recs]
    done = [
        sim.finish_at[r.session_id, r.turn] - r.arrival_s
        for r in recs
        if (r.session_id, r.turn) in sim.finish_at
    ]
    by_size = sorted(recs, key=lambda r: r.prompt_tokens)
    top = by_size[-max(1, len(recs) // 10) :]
    bottom = by_size[: len(recs) // 2]
    return {
        **{k: job[k] for k in ("trace", "arm", "rate", "cpu_blocks", "sec_blocks")},
        "p90": _pct(ttft, 90),
        **_stats("ttft", ttft),
        **_stats("done", done),
        **_stats("big_ttft", [r.ttft_s for r in top]),
        **_stats("small_ttft", [r.ttft_s for r in bottom]),
        "turns": len(recs),
        "queue_s": sum(r.queue_s for r in recs),
        "prefill_s": sum(r.prefill_s for r in recs),
        "load_s": sum(r.load_s for r in recs),
        "preemptions": result.link.get("preemptions", 0),
        "swap_reload_blocks": result.link.get("swap_reload_blocks", 0),
        "swap_miss_tokens": result.link.get("swap_miss_tokens", 0),
        "miss_tokens": sum(r.miss_blocks for r in recs) * job["block_tokens"],
        "sim_end_s": result.link.get("sim_end_s", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("traces", nargs="+")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--gpu-blocks", type=int, default=66500)
    p.add_argument("--cpu-blocks", type=int, default=140000)
    p.add_argument("--sec-blocks", type=int, default=0)
    p.add_argument("--block-tokens", type=int, default=256)
    p.add_argument("--cost-json", required=True)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    jobs = [
        {
            "trace": t,
            "arm": a,
            "rate": r,
            "gpu_blocks": args.gpu_blocks,
            "cpu_blocks": args.cpu_blocks,
            "sec_blocks": args.sec_blocks,
            "block_tokens": args.block_tokens,
            "cost_json": args.cost_json,
        }
        for t in args.traces
        for a in args.arms
        for r in args.rates
    ]
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "w") as f:
        for row in pool.map(run_one, jobs):
            print(
                f"{row['trace'].split('/')[-1]} {row['arm']} r={row['rate']} "
                f"ttft p90={row['ttft_90']:.2f} p99={row['ttft_99']:.2f} "
                f"done p90={row['done_90']:.1f} swaps={row['preemptions']}",
                flush=True,
            )
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
