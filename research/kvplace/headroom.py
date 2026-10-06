"""Where does resumption latency go, and how much can any placement or
restore policy recover?

`breakdown` splits resumption TTFT into GPU queueing, tier loads, waits for
in-flight warm-ups, and prefill of new tokens, for the median and the tail.
It also reports a perfect-cache bound: an unbounded GPU, so every reusable
block is a GPU hit. No placement or restore policy can beat it.

`map` sweeps tier sizes and link bandwidths and reports, per configuration,
the p90 gain over LRU of the best oracle policy and of the bound.

Example:
    python -m kvplace.headroom breakdown traces/tl300_full_r05_s1.jsonl
    python -m kvplace.headroom map traces/tl300_full_r05_s1.jsonl \
        --out results/headroom_map.json

"""

import argparse
import itertools
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from kvplace.sim import CostModel, Simulator, _pct, get_policy
from kvplace.trace import read_trace

UNBOUNDED = 10**8
SEC_BLOCKS = 10**7
COMPONENTS = ("queue_s", "load_s", "warmup_wait_s", "prefill_s")
GAP_BUCKETS = ((0, 1), (1, 10), (10, 60), (60, float("inf")))
ORACLE_ARMS = (("reuse", "oracle"), ("warm:2", "oracle"))


def _run(job: tuple) -> dict:
    trace, policy, predictor, gpu, cpu, cost = job
    result = Simulator(
        read_trace(trace), get_policy(policy), predictor, cost, gpu, cpu, SEC_BLOCKS
    ).run()
    resume = [r for r in result.records if r.turn > 0]
    return {
        "policy": policy,
        "predictor": predictor,
        "gpu_blocks": gpu,
        "cpu_blocks": cpu,
        "cost": cost.__dict__,
        "ttft": [r.ttft_s for r in resume],
        "gap": [r.gap_s for r in resume],
        "components": {c: [getattr(r, c) for r in resume] for c in COMPONENTS},
        "new_tokens": [
            r.prompt_tokens - r.reusable_blocks * cost.block_tokens for r in resume
        ],
    }


def _parallel(jobs: list[tuple]) -> list[dict]:
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        return list(ex.map(_run, jobs))


def _tail_shares(run: dict, lo: float, hi: float) -> dict:
    """Mean of each component over turns whose TTFT percentile is in [lo, hi)."""
    ttft = run["ttft"]
    a, b = _pct(ttft, lo), _pct(ttft, hi) if hi < 100 else float("inf")
    idx = [i for i, t in enumerate(ttft) if a <= t < b or (hi == 100 and t >= a)]
    out = {"turns": len(idx), "ttft": statistics.fmean(ttft[i] for i in idx)}
    for c in COMPONENTS:
        out[c] = statistics.fmean(run["components"][c][i] for i in idx)
    out["new_tokens"] = statistics.fmean(run["new_tokens"][i] for i in idx)
    return out


def breakdown(args) -> dict:
    cost = CostModel()
    arms = [("lru", "none", args.gpu_blocks), ("warm:inf", "none", args.gpu_blocks)]
    arms += [(p, pred, args.gpu_blocks) for p, pred in ORACLE_ARMS]
    arms += [("lru", "none", UNBOUNDED)]
    jobs = [
        (args.trace, pol, pred, gpu, cpu, cost)
        for cpu in args.cpu_blocks
        for pol, pred, gpu in arms
    ]
    runs = _parallel(jobs)
    out = []
    for run in runs:
        label = (
            "bound(perfect cache)"
            if run["gpu_blocks"] == UNBOUNDED
            else f"{run['policy']}/{run['predictor']}"
        )
        ttft = run["ttft"]
        row = {
            "cpu_blocks": run["cpu_blocks"],
            "arm": label,
            "p50": _pct(ttft, 50),
            "p90": _pct(ttft, 90),
            "p99": _pct(ttft, 99),
            "bands": {
                "p0-50": _tail_shares(run, 0, 50),
                "p90-99": _tail_shares(run, 90, 99),
                "p99+": _tail_shares(run, 99, 100),
            },
            "by_gap": {},
        }
        for lo, hi in GAP_BUCKETS:
            sel = [t for t, g in zip(ttft, run["gap"]) if lo <= g < hi]
            row["by_gap"][f"{lo}-{hi}s"] = {
                "turns": len(sel),
                "mean_ttft": statistics.fmean(sel) if sel else 0.0,
            }
        out.append(row)

    print("cpu\tarm\tp50\tp90\tp99")
    for r in out:
        print(
            f"{r['cpu_blocks']}\t{r['arm']}\t{r['p50']:.3f}\t{r['p90']:.3f}\t"
            f"{r['p99']:.3f}"
        )
    print("\nMean TTFT components (s) by TTFT band:")
    print("cpu\tarm\tband\tturns\tttft\tqueue\tload\twarm_wait\tprefill\tnew_tok")
    for r in out:
        for band, b in r["bands"].items():
            print(
                f"{r['cpu_blocks']}\t{r['arm']}\t{band}\t{b['turns']}\t"
                f"{b['ttft']:.3f}\t{b['queue_s']:.3f}\t{b['load_s']:.3f}\t"
                f"{b['warmup_wait_s']:.3f}\t{b['prefill_s']:.3f}\t"
                f"{b['new_tokens']:.0f}"
            )
    print("\nMean resumption TTFT (s) by preceding gap:")
    buckets = [f"{lo}-{hi}s" for lo, hi in GAP_BUCKETS]
    print("cpu\tarm\t" + "\t".join(buckets))
    for r in out:
        cells = [
            f"{r['by_gap'][b]['mean_ttft']:.3f} (n={r['by_gap'][b]['turns']})"
            for b in buckets
        ]
        print(f"{r['cpu_blocks']}\t{r['arm']}\t" + "\t".join(cells))
    return {"breakdown": out}


def headroom_map(args) -> dict:
    base = CostModel()
    configs = [
        (gpu, gpu * ratio, replace(base, gpu_cpu_gbps=g2c, sec_cpu_gbps=s2c))
        for gpu, ratio, g2c, s2c in itertools.product(
            args.gpu_grid, args.cpu_ratios, args.gpu_cpu_gbps, args.sec_cpu_gbps
        )
    ]
    jobs = []
    for gpu, cpu, cost in configs:
        jobs.append((args.trace, "lru", "none", gpu, cpu, cost))
        jobs += [(args.trace, p, pred, gpu, cpu, cost) for p, pred in ORACLE_ARMS]
    # The bound depends only on prefill and queueing, not on tiers or links.
    jobs.append((args.trace, "lru", "none", UNBOUNDED, 0, base))
    runs = _parallel(jobs)
    bound_p90 = _pct(runs[-1]["ttft"], 90)

    def p90(run):
        return _pct(run["ttft"], 90)

    out = []
    per = 1 + len(ORACLE_ARMS)
    for i, (gpu, cpu, cost) in enumerate(configs):
        lru, *oracles = runs[i * per : (i + 1) * per]
        best = min(oracles, key=p90)
        out.append(
            {
                "gpu_blocks": gpu,
                "cpu_blocks": cpu,
                "gpu_cpu_gbps": cost.gpu_cpu_gbps,
                "sec_cpu_gbps": cost.sec_cpu_gbps,
                "lru_p90": p90(lru),
                "oracle_p90": p90(best),
                "oracle_arm": best["policy"],
                "oracle_gain": 1 - p90(best) / p90(lru),
                "bound_gain": 1 - bound_p90 / p90(lru),
            }
        )
    print(f"perfect-cache bound p90 = {bound_p90:.3f}s")
    print("gpu\tcpu\tg2c\ts2c\tlru_p90\toracle_p90\toracle_gain\tbound_gain\tarm")
    for r in out:
        print(
            f"{r['gpu_blocks']}\t{r['cpu_blocks']}\t{r['gpu_cpu_gbps']:g}\t"
            f"{r['sec_cpu_gbps']:g}\t{r['lru_p90']:.3f}\t{r['oracle_p90']:.3f}\t"
            f"{r['oracle_gain']:+.1%}\t{r['bound_gain']:+.1%}\t{r['oracle_arm']}"
        )
    return {"bound_p90": bound_p90, "map": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("breakdown")
    b.add_argument("trace")
    b.add_argument("--gpu-blocks", type=int, default=12000)
    b.add_argument("--cpu-blocks", type=int, nargs="+", default=[24000, 96000])
    b.add_argument("--out")
    m = sub.add_parser("map")
    m.add_argument("trace")
    m.add_argument("--gpu-grid", type=int, nargs="+", default=[6000, 12000, 24000])
    m.add_argument("--cpu-ratios", type=int, nargs="+", default=[2, 8])
    m.add_argument("--gpu-cpu-gbps", type=float, nargs="+", default=[25.0])
    m.add_argument("--sec-cpu-gbps", type=float, nargs="+", default=[3.0, 7.0])
    m.add_argument("--out")
    args = p.parse_args()
    result = breakdown(args) if args.cmd == "breakdown" else headroom_map(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), **result}, f)


if __name__ == "__main__":
    main()
