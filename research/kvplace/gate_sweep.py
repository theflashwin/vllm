"""Does agent-aware CPU eviction gain capacity at any sustainable load?

LRU vs the reuse policy, 7B (measured costs) and 70B (costs scaled, see `llama70`).
Hints are free server-side updates, which favors the policy.

Example:
    python -m kvplace.gate_sweep --out results/gate/gate.jsonl
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from kvplace.sim import CostModel, Simulator, _pct, get_policy
from kvplace.trace import read_trace

GB = 10**9
GPUS_70B = 8
KV_70B = 327680  # Llama-3-70B bf16 shape: 2*80*8*128*2
FLOPS_70B = 10.0

TRACES = {
    "tl_s1": "traces/fx/tl300_sp_s1.jsonl",
    "tl_s3": "traces/fx/tl300_sp_s3.jsonl",
    "cat": "traces/fx/cat.jsonl",
}
RATES = {
    ("7B", "tl"): [0.003, 0.004, 0.0045, 0.005, 0.0055, 0.0065],
    ("7B", "cat"): [0.006, 0.009, 0.011, 0.0125, 0.014, 0.017],
    ("70B", "tl"): [0.002, 0.003, 0.0045, 0.006, 0.008, 0.012],
    ("70B", "cat"): [0.005, 0.0075, 0.011, 0.015, 0.02, 0.03],
}
CONFIGS = [
    ("7B", 7, 0.96),
    ("7B", 32, 0.96),
    ("7B", 7, 3.0),
    ("70B", 256, 0.96),
    ("70B", 1024, 0.96),
    ("70B", 256, 3.0),
]


def llama70(c: CostModel) -> CostModel:
    """An 8-GPU node serving a 70B model: estimated, not measured.

    Compute scales with FLOPS_70B over GPUS_70B; scans and links with KV bytes
    over GPUS_70B.
    """
    speed = GPUS_70B / FLOPS_70B
    scan = (KV_70B / c.kv_bytes_per_token) / GPUS_70B
    return replace(
        c,
        kv_bytes_per_token=KV_70B,
        gpu_cpu_gbps=c.gpu_cpu_gbps * GPUS_70B,
        prefill_tok_per_s=c.prefill_tok_per_s * speed,
        decode_s_per_tok=c.decode_s_per_tok / speed,
        decode_s_per_ctx_tok=c.decode_s_per_ctx_tok * scan,
        prefill_s_per_tok_per_ctx=c.prefill_s_per_tok_per_ctx * scan,
    )


def run_one(job: dict) -> dict:
    sessions = read_trace(TRACES[job["trace"]])
    span = max(s.start_s for s in sessions)
    factor = (len(sessions) / job["rate"]) / span
    for s in sessions:
        s.start_s *= factor
    cost = CostModel.from_json("costs_scan_a100.json")
    gpu_gb = 61
    if job["model"] == "70B":
        cost = llama70(cost)
        gpu_gb = 61 * GPUS_70B
    cost = replace(cost, block_tokens=256, sec_cpu_gbps=job["disk_gbps"])
    blocks = lambda gb: int(gb * GB / cost.block_bytes)  # noqa: E731
    policy = "lru" if job["arm"] == "lru" else "reuse_evict"
    sim = Simulator(
        sessions,
        get_policy(policy),
        "none" if job["arm"] == "lru" else "tool-nfh2",
        cost,
        blocks(gpu_gb),
        blocks(job["cpu_gb"]),
        10_000_000,
        contend_sec=True,
        write_model="shared",
        gpu_admission=True,
        server_hints=True,
    )
    recs = sim.run().records
    resumes = [r for r in recs if r.turn > 0]
    ttft = [r.ttft_s for r in recs]
    return {
        **job,
        "p50": _pct(ttft, 50),
        "p90": _pct(ttft, 90),
        "p99": _pct(ttft, 99),
        "resume_p90": _pct([r.ttft_s for r in resumes], 90),
        "disk_resume_share": sum(r.sec_hit_blocks > 0 for r in resumes) / len(resumes),
        "disk_gb": sum(r.sec_hit_blocks for r in recs) * cost.block_bytes / GB,
        "miss_tokens": sum(r.miss_blocks for r in recs) * cost.block_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    jobs = [
        dict(
            model=m,
            cpu_gb=cpu,
            disk_gbps=disk,
            trace=t,
            arm=a,
            rate=r,
        )
        for m, cpu, disk in CONFIGS
        for t in TRACES
        for a in ("lru", "ours")
        for r in RATES[m, t.split("_")[0]]
    ]
    print(len(jobs), flush=True)
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "w") as f:
        for row in pool.map(run_one, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
