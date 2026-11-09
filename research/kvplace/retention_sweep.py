"""Fleet effect of client retention: a share of sessions adopts a cache plan.

Everyone else keeps the 5-minute tier. Reports TTFT for adopters and for others,
misses, pings, and time-integrated memory per tier.

Example:
    python -m kvplace.retention_sweep traces/fx/tl300_sp_s1.jsonl \
        --arms base ka1h@0.5 ttl1h@1 --rates 0.005 0.01 \
        --cost-json costs_scan_a100.json --out fleet.json

"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from kvplace.sim import CostModel, Retention, Simulator, _pct, get_policy
from kvplace.trace import read_trace

DEFAULT = Retention()
PLANS = {
    "base": DEFAULT,
    "lru": Retention(ttl_s=float("inf")),
    "ttl30m": Retention(ttl_s=1800.0),
    "ttl1h": Retention(ttl_s=3600.0),
    "ka30m": Retention(ping_s=270.0, ping_cap_s=1800.0),
    "ka1h": Retention(ping_s=270.0, ping_cap_s=3600.0),
    "ka1h_notail": Retention(ping_s=270.0, ping_cap_s=3600.0, ping_tail=False),
    "ka1h_touch": Retention(ping_s=270.0, ping_cap_s=3600.0, ping_touch=True),
}


def adopters(session_ids: list[str], share: float) -> set[str]:
    """A stable order, so a larger share contains every smaller one."""
    order = sorted(session_ids, key=lambda sid: hashlib.md5(sid.encode()).hexdigest())
    return set(order[: round(share * len(order))])


def _group_stats(name: str, records: list, block_tokens: int) -> dict:
    resumes = [r for r in records if r.turn > 0]
    prompt = sum(r.prompt_tokens for r in resumes)
    return {
        f"{name}_p50": _pct([r.ttft_s for r in records], 50),
        f"{name}_p90": _pct([r.ttft_s for r in records], 90),
        f"{name}_p99": _pct([r.ttft_s for r in records], 99),
        f"{name}_resume_miss": sum(r.miss_blocks for r in resumes)
        * block_tokens
        / max(1, prompt),
    }


def run_one(job: dict) -> dict:
    sessions = read_trace(job["trace"])
    span = max(s.start_s for s in sessions)
    factor = (len(sessions) / job["rate"]) / span
    for s in sessions:
        s.start_s *= factor
    plan_name, _, share = job["arm"].partition("@")
    chosen = adopters([s.session_id for s in sessions], float(share or 1))
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
        retention={
            s.session_id: PLANS[plan_name] if s.session_id in chosen else DEFAULT
            for s in sessions
        },
    )
    result = sim.run()
    gb_per_block = job["block_tokens"] * cost.kv_bytes_per_token / 1e9
    end_s = result.link["sim_end_s"]
    row = {
        **{k: job[k] for k in ("trace", "arm", "rate", "cpu_blocks", "sec_blocks")},
        "share": len(chosen) / len(sessions),
        "sim_end_s": end_s,
        "pings": sim.pings,
        "ping_tokens": sim.ping_tokens,
        "prompt_tokens": sum(r.prompt_tokens for r in result.records),
        "miss_tokens": sum(r.miss_blocks for r in result.records) * job["block_tokens"],
        "reload_tokens": sum(
            r.cpu_hit_blocks + r.sec_hit_blocks for r in result.records
        )
        * job["block_tokens"],
        "gpu_hit_tokens": sum(r.gpu_hit_blocks for r in result.records)
        * job["block_tokens"],
    }
    for tier, block_s in sim.occupancy_block_s.items():
        row[f"{tier}_gb_h"] = block_s * gb_per_block / 3600
        row[f"{tier}_gb_mean"] = block_s * gb_per_block / max(end_s, 1e-9)
    row.update(_group_stats("all", result.records, job["block_tokens"]))
    for name, group in (
        ("adopter", [r for r in result.records if r.session_id in chosen]),
        ("other", [r for r in result.records if r.session_id not in chosen]),
    ):
        if group:
            row.update(_group_stats(name, group, job["block_tokens"]))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", nargs="+")
    p.add_argument("--arms", nargs="+", required=True, help="plan[@share]")
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--gpu-blocks", type=int, default=66500)
    p.add_argument("--cpu-blocks", type=int, default=140000)
    p.add_argument("--sec-blocks", type=int, default=10_000_000)
    p.add_argument("--block-tokens", type=int, default=256)
    p.add_argument("--cost-json", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out")
    args = p.parse_args()
    jobs = [
        {
            "trace": t,
            "arm": arm,
            "rate": rate,
            "gpu_blocks": args.gpu_blocks,
            "cpu_blocks": args.cpu_blocks,
            "sec_blocks": args.sec_blocks,
            "block_tokens": args.block_tokens,
            "cost_json": args.cost_json,
        }
        for t in args.traces
        for rate in args.rates
        for arm in args.arms
    ]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(run_one, jobs):
            rows.append(row)
            print(
                f"{row['trace'].split('/')[-1]} {row['arm']} r={row['rate']} "
                f"other p90={row.get('other_p90', float('nan')):.2f} "
                f"all p90={row['all_p90']:.2f} "
                f"miss={row['all_resume_miss']:.3f} pings={row['pings']} "
                f"cpu={row['cpu_gb_mean']:.0f}GB sec={row['sec_gb_mean']:.0f}GB",
                flush=True,
            )
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
