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

from kvplace.sim import CostModel, Simulator, get_policy, measured_sec_write_gbps
from kvplace.trace import read_trace

METRICS = (
    "resume_ttft_p50",
    "resume_ttft_p90",
    "resume_ttft_p99",
    "sec_to_cpu_prefetch_blocks",
    "prefetch_wasted_blocks",
)


def _run(job: tuple) -> dict:
    """job = (trace, gpu, cpu, sec_gbps, policy, predictor, cost_json[, opts]).

    Optional `opts` (dict) enables the disk write model:
        write_model         off | shared | separate (default off)
        write_interference  read slowdown while writing ("separate")
        sec_write_gbps      write BW; default: fitted from cost_json's
                            `_fs_rows` when the read BW is also the measured
                            one (sec_gbps is None), else equal to read BW
    """
    trace, gpu, cpu, sec_gbps, policy, predictor, cost_json, *rest = job
    opts = rest[0] if rest else {}
    cost = CostModel.from_json(cost_json) if cost_json else CostModel()
    if sec_gbps is not None:
        cost = replace(cost, sec_cpu_gbps=sec_gbps)
    write_model = opts.get("write_model", "off")
    if write_model != "off":
        wbw = opts.get("sec_write_gbps")
        if wbw is None and cost_json and sec_gbps is None:
            wbw = measured_sec_write_gbps(cost_json)
        cost = replace(cost, sec_write_gbps=wbw)
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
            write_model=write_model,
            write_interference=opts.get("write_interference", 0.0),
        )
        .run()
        .summary()
    )
    row = {
        "gpu_blocks": gpu,
        "cpu_blocks": cpu,
        "sec_cpu_gbps": cost.sec_cpu_gbps,
        "policy": policy,
        "predictor": predictor,
        **{m: summary[m] for m in METRICS},
    }
    if write_model != "off":
        row |= {
            "write_model": write_model,
            "write_interference": opts.get("write_interference", 0.0),
        }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--gpu-blocks", type=int, default=12000)
    p.add_argument(
        "--configs",
        nargs="+",
        default=["24000:7", "96000:3", "96000:7"],
        help="cpu_blocks:secondary_read_gbps ('m' keeps the measured value)",
    )
    p.add_argument("--cost-json", help="CostModel overrides (microbench output)")
    p.add_argument(
        "--policies", nargs="+", default=["gpu_aware", "reuse_evict", "gpu_cpu_aware"]
    )
    p.add_argument("--predictors", nargs="+", default=["oracle", "noisy:1.0"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--write-model", choices=["off", "shared", "separate"], default="off"
    )
    p.add_argument("--write-interference", type=float, default=0.0)
    p.add_argument("--out")
    args = p.parse_args()
    write = (
        (
            {
                "write_model": args.write_model,
                "write_interference": args.write_interference,
            },
        )
        if args.write_model != "off"
        else ()
    )

    jobs = []
    for config in args.configs:
        cpu, sec_gbps = config.split(":")
        sec = None if sec_gbps == "m" else float(sec_gbps)
        base = (args.trace, args.gpu_blocks, int(cpu), sec)
        # LRU gets each predictor too: post-response predictors add hint-only
        # requests, which LRU must also see for a matched comparison.
        jobs += [
            (*base, pol, pred, args.cost_json, *write)
            for pol in ("lru", *args.policies)
            for pred in args.predictors
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
