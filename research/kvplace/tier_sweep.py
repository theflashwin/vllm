"""Where should agent awareness live? Compare placement policies with a shared
(contended) secondary->CPU read link, so prefetch competes with demand loads.

Policies (see `kvplace.sim.POLICIES`):
    lru            agent-blind everywhere
    gpu_aware      agent-aware GPU<->CPU eviction only (CacheWise/TokenCake-like)
    reuse_evict    agent-aware CPU->disk demotion only
    gpu_cpu_aware  agent-aware at both boundaries
    reuse          CPU demotion + disk->CPU prefetch + selective writes

Example:
    python -m kvplace.tier_sweep traces/tl300_sp_s1.jsonl \
        --configs 24000:7 96000:3 96000:7 \
        --policies gpu_aware reuse_evict gpu_cpu_aware \
        --out results/tier_sweep.json

"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from kvplace.sim import CostModel, Simulator, get_policy
from kvplace.trace import read_trace

METRICS = (
    "resume_ttft_p50",
    "resume_ttft_p90",
    "resume_ttft_p99",
    "sec_to_cpu_prefetch_blocks",
    "prefetch_wasted_blocks",
)


def _run(job: tuple) -> dict:
    trace, gpu, cpu, sec_gbps, policy, predictor = job
    cost = replace(CostModel(), sec_cpu_gbps=sec_gbps)
    summary = (
        Simulator(
            read_trace(trace),
            get_policy(policy),
            predictor,
            cost,
            gpu,
            cpu,
            10**7,
            contend_sec=True,
        )
        .run()
        .summary()
    )
    return {
        "gpu_blocks": gpu,
        "cpu_blocks": cpu,
        "sec_cpu_gbps": sec_gbps,
        "policy": policy,
        "predictor": predictor,
        **{m: summary[m] for m in METRICS},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--gpu-blocks", type=int, default=12000)
    p.add_argument(
        "--configs",
        nargs="+",
        default=["24000:7", "96000:3", "96000:7"],
        help="cpu_blocks:secondary_read_gbps",
    )
    p.add_argument(
        "--policies", nargs="+", default=["gpu_aware", "reuse_evict", "gpu_cpu_aware"]
    )
    p.add_argument("--predictors", nargs="+", default=["oracle", "noisy:1.0"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out")
    args = p.parse_args()

    jobs = []
    for config in args.configs:
        cpu, sec_gbps = config.split(":")
        base = (args.trace, args.gpu_blocks, int(cpu), float(sec_gbps))
        jobs.append((*base, "lru", "none"))
        jobs += [
            (*base, pol, pred) for pol in args.policies for pred in args.predictors
        ]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_run, jobs))

    print("cpu\tsec_gbps\tpolicy\tpredictor\tp50\tp90\tp99\tprefetch\twasted")
    for r in rows:
        print(
            f"{r['cpu_blocks']}\t{r['sec_cpu_gbps']:g}\t{r['policy']}\t"
            f"{r['predictor']}\t{r['resume_ttft_p50']:.3f}\t"
            f"{r['resume_ttft_p90']:.3f}\t{r['resume_ttft_p99']:.3f}\t"
            f"{r['sec_to_cpu_prefetch_blocks']}\t{r['prefetch_wasted_blocks']}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f)


if __name__ == "__main__":
    main()
