"""Repeat the headline comparisons over several trace samples (seeds) and
report mean and standard deviation of p90 resumption TTFT and of the gain
over the GPU-only agent-aware baseline.

Settings match the midpoint slides. Traces per seed (from kvplace.tracelab):
    traces/tl300_sp_s<seed>.jsonl        0.05 sessions/s
    traces/tl300_sp_s<seed>_r0.01.jsonl  0.01 sessions/s (L4 setting)

Example:
    python -m kvplace.multiseed --seeds 1 2 3 4 5 --out results/multiseed.json

"""

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor

from kvplace.tier_sweep import _run

POLICIES = ("lru", "gpu_aware", "gpu_cpu_aware")
# name -> (trace pattern, gpu blocks, cpu blocks, secondary GB/s, cost json,
#          predictors)
SETTINGS = {
    "l4_slow_disk": (
        "traces/tl300_sp_s{seed}_r0.01.jsonl",
        8000,
        48000,
        None,
        "costs_l4.json",
        ("tool-nfh", "tool"),
    ),
    "small_cpu_7gbps": (
        "traces/tl300_sp_s{seed}.jsonl",
        12000,
        24000,
        7.0,
        None,
        ("tool-nfh", "tool"),
    ),
    "large_cpu_3gbps": (
        "traces/tl300_sp_s{seed}.jsonl",
        12000,
        96000,
        3.0,
        None,
        ("tool-nfh", "tool", "oracle", "noisy:1.0"),
    ),
    "h100_size": (
        "traces/tl300_sp_s{seed}.jsonl",
        65000,
        260000,
        3.0,
        None,
        ("tool-nfh", "tool"),
    ),
}


def _job(args: tuple) -> dict:
    setting, seed, policy, predictor = args
    trace, gpu, cpu, sec, cost_json, _ = SETTINGS[setting]
    row = _run((trace.format(seed=seed), gpu, cpu, sec, policy, predictor, cost_json))
    return {"setting": setting, "seed": seed, **row}


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out")
    args = p.parse_args()

    jobs = [
        (s, seed, pol, pred)
        for s in args.settings
        for seed in args.seeds
        for pred in SETTINGS[s][5]
        for pol in POLICIES
    ]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_job, jobs))

    def p90(s, seed, pol, pred):
        return next(
            r["resume_ttft_p90"]
            for r in rows
            if (r["setting"], r["seed"], r["policy"], r["predictor"])
            == (s, seed, pol, pred)
        )

    summary = []
    print("setting\tpredictor\tlru\tgpu_aware\tours\tgain_vs_gpu_aware")
    for s in args.settings:
        for pred in SETTINGS[s][5]:
            vals = {
                pol: [p90(s, seed, pol, pred) for seed in args.seeds]
                for pol in POLICIES
            }
            gains = [
                o / g - 1 for o, g in zip(vals["gpu_cpu_aware"], vals["gpu_aware"])
            ]
            entry = {
                "setting": s,
                "predictor": pred,
                **{pol: _mean_sd(v) for pol, v in vals.items()},
                "gain": _mean_sd(gains),
                "seeds": args.seeds,
            }
            summary.append(entry)
            cells = [f"{entry[pol][0]:.2f}±{entry[pol][1]:.2f}" for pol in POLICIES]
            g, gsd = entry["gain"]
            print(f"{s}\t{pred}\t" + "\t".join(cells) + f"\t{g:+.1%}±{gsd:.1%}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": summary}, f)


if __name__ == "__main__":
    main()
