"""Evaluation protocol: paired per-seed gains with bootstrap CIs on
developer-held-out traces, plus a knob sensitivity sweep.

For each headline setting (`kvplace.multiseed.SETTINGS`), seed, policy
(lru / gpu_aware / gpu_cpu_aware), predictor and default horizon, runs the
simulator and keeps the full `Simulator.summary()` (TTFT percentiles, tier
mix `resume_frac_*`, disk reads `sec_to_cpu_demand_blocks`, ...). The gain of
a seed is paired: p90(gpu_cpu_aware) / p90(gpu_aware) - 1 on the same trace
with the same predictor. Reported: mean, median, a 95% percentile-bootstrap
CI of the mean over seeds, and wins (gain < 0).

Traces: `<trace-dir>/<prefix>_s<seed>.jsonl` (0.05 sessions/s) and
`<prefix>_s<seed>_r0.01.jsonl` (0.01/s, the L4 setting), from
`kvplace.tracelab --user-split {train,test}`.

`--human-quantiles` rewrites `tool-nfh`/`tool-nfh2` predictors as
`<name>:200:<q>`; `--horizons` sets `Simulator(default_horizon_s=...)`.
Finished runs are cached in `--cache` (JSONL) and reused on rerun.

Examples:
    # held-out evaluation
    python -m kvplace.evalproto --trace-dir traces --prefix test \
        --seeds $(seq 1 20) --predictors tool-nfh2 --out results/eval_test.json
    # sensitivity sweep on the train split (no LRU needed for the gain)
    python -m kvplace.evalproto --trace-dir traces --prefix train \
        --settings large_cpu_3gbps small_cpu_7gbps --seeds $(seq 1 10) \
        --policies gpu_aware gpu_cpu_aware --predictors tool-nfh2 \
        --horizons 10 30 120 --human-quantiles 0.5 0.75 0.9 0.99 \
        --out results/sweep_train.json

"""

import argparse
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from kvplace.multiseed import SETTINGS
from kvplace.sim import CostModel, Simulator, get_policy
from kvplace.trace import read_trace

RESEARCH = Path(__file__).resolve().parent.parent
POLICIES = ("lru", "gpu_aware", "gpu_cpu_aware")
KEEP = (
    "resume_ttft_mean",
    "resume_ttft_p50",
    "resume_ttft_p90",
    "resume_ttft_p99",
    "first_ttft_p50",
    "resume_frac_gpu",
    "resume_frac_cpu",
    "resume_frac_sec",
    "resume_frac_lost",
    "sec_to_cpu_demand_blocks",
    "sec_to_cpu_prefetch_blocks",
    "cpu_to_sec_write_blocks",
    "hint_requests",
    "turns",
)


def trace_path(trace_dir: str, prefix: str, setting: str, seed: int) -> str:
    sfx = "_r0.01" if "_r0.01" in SETTINGS[setting][0] else ""
    return os.path.join(trace_dir, f"{prefix}_s{seed}{sfx}.jsonl")


def predictor_name(pred: str, human_q: float | None) -> str:
    if human_q is None:
        return pred
    if not pred.split(":")[0].endswith(("-nfh", "-nfh2")):
        raise ValueError(f"--human-quantiles needs a tool-nfh* predictor, got {pred}")
    window = pred.partition(":")[2] or "200"
    return f"{pred.split(':')[0]}:{window}:{human_q:g}"


def run_job(job: dict) -> dict:
    _, gpu, cpu, sec, cost_json, _ = SETTINGS[job["setting"]]
    cost = CostModel.from_json(str(RESEARCH / cost_json)) if cost_json else CostModel()
    if sec is not None:
        cost = replace(cost, sec_cpu_gbps=sec)
    summary = (
        Simulator(
            read_trace(job["trace"]),
            get_policy(job["policy"]),
            job["predictor"],
            cost,
            gpu,
            cpu,
            10**7,
            contend_sec=True,
            default_horizon_s=job["horizon"],
            write_model=job.get("write_model", "off"),
            write_interference=job.get("write_interference", 0.0),
        )
        .run()
        .summary()
    )
    return {**job, **{m: summary.get(m, 0.0) for m in KEEP}}


def _key(job: dict) -> str:
    key = [job[k] for k in ("setting", "trace", "policy", "predictor", "horizon")]
    # Write-model fields join the key only when set, so old caches still match.
    if job.get("write_model", "off") != "off":
        key += [job["write_model"], job.get("write_interference", 0.0)]
    return json.dumps(key)


def bootstrap_ci(
    xs: list[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling seeds)."""
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()), float(a.mean())) if len(a) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_gain(
    rows: list[dict],
    metric: str = "resume_ttft_p90",
    n_boot: int = 10_000,
    base: str = "gpu_aware",
    ours: str = "gpu_cpu_aware",
) -> dict:
    """rows: one group (setting, predictor, horizon) over seeds."""
    by = {(r["seed"], r["policy"]): r for r in rows}
    seeds = sorted({s for s, p in by if (s, base) in by and (s, ours) in by})
    gains = [by[(s, ours)][metric] / by[(s, base)][metric] - 1 for s in seeds]
    if not gains:
        return {}
    lo, hi = bootstrap_ci(gains, n_boot)
    return {
        "metric": metric,
        "n": len(gains),
        "mean": statistics.fmean(gains),
        "median": statistics.median(gains),
        "ci95": [lo, hi],
        "wins": sum(g < 0 for g in gains),
        "per_seed": dict(zip(seeds, gains)),
    }


def summarize(
    rows: list[dict],
    n_boot: int = 10_000,
    base: str = "gpu_aware",
    ours: str = "gpu_cpu_aware",
) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["setting"], r["predictor"], r["horizon"]), []).append(r)
    out = []
    for (setting, pred, horizon), rs in sorted(groups.items()):
        per_policy = {}
        for pol in sorted({r["policy"] for r in rs}):
            prs = [r for r in rs if r["policy"] == pol]
            per_policy[pol] = {m: statistics.fmean(r[m] for r in prs) for m in KEEP}
            per_policy[pol]["n"] = len(prs)
        out.append(
            {
                "setting": setting,
                "predictor": pred,
                "horizon": horizon,
                "gain_p90": paired_gain(rs, "resume_ttft_p90", n_boot, base, ours),
                "gain_p50": paired_gain(rs, "resume_ttft_p50", n_boot, base, ours),
                "gain_p99": paired_gain(rs, "resume_ttft_p99", n_boot, base, ours),
                "policies": per_policy,
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trace-dir", default="traces")
    p.add_argument(
        "--prefix", default="test", help="trace file prefix (test/train/tl300_sp)"
    )
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    p.add_argument(
        "--seeds-per-setting",
        nargs="*",
        default=[],
        help="override as setting=N (use seeds 1..N), e.g. h100_size=10",
    )
    p.add_argument("--policies", nargs="+", default=list(POLICIES))
    p.add_argument("--base-policy", default="gpu_aware", help="paired-gain baseline")
    p.add_argument("--ours-policy", default="gpu_cpu_aware", help="paired-gain policy")
    p.add_argument("--write-model", default="off", help="off | shared | separate")
    p.add_argument("--write-interference", type=float, default=0.0)
    p.add_argument("--predictors", nargs="+", default=["tool-nfh2"])
    p.add_argument("--horizons", type=float, nargs="+", default=[30.0])
    p.add_argument("--human-quantiles", type=float, nargs="+", default=[None])
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cache", help="JSONL of finished runs (appended, reused)")
    p.add_argument("--out")
    args = p.parse_args()

    per_setting = dict(x.split("=") for x in args.seeds_per_setting)
    jobs = []
    for s in args.settings:
        seeds = args.seeds
        if s in per_setting:
            seeds = [x for x in seeds if x <= int(per_setting[s])]
        for seed in seeds:
            for pred in args.predictors:
                for hq in args.human_quantiles:
                    for h in args.horizons:
                        for pol in args.policies:
                            jobs.append(
                                {
                                    "setting": s,
                                    "seed": seed,
                                    "trace": trace_path(
                                        args.trace_dir, args.prefix, s, seed
                                    ),
                                    "policy": pol,
                                    "predictor": predictor_name(pred, hq),
                                    "horizon": h,
                                    "write_model": args.write_model,
                                    "write_interference": args.write_interference,
                                }
                            )
    done: dict[str, dict] = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            for line in f:
                r = json.loads(line)
                done[_key(r)] = r
    todo = [j for j in jobs if _key(j) not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} cached", flush=True)
    # Slowest settings first so the pool drains evenly.
    order = {
        "h100_size": 0,
        "l4_slow_disk": 1,
        "large_cpu_3gbps": 2,
        "small_cpu_7gbps": 3,
    }
    todo.sort(key=lambda j: order.get(j["setting"], 9))
    cache = open(args.cache, "a") if args.cache else None
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done[_key(r)] = r
            if cache:
                cache.write(json.dumps(r) + "\n")
                cache.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done", flush=True)
    if cache:
        cache.close()
    rows = [done[_key(j)] for j in jobs]
    base, ours = args.base_policy, args.ours_policy
    summary = summarize(rows, args.n_boot, base, ours)

    print(
        "setting\tpredictor\thorizon\tn\tgpu_aware_p90\tours_p90\t"
        "gain_mean\tgain_median\tci95\twins"
    )
    for e in summary:
        g = e["gain_p90"]
        if not g:
            continue
        pp = e["policies"]
        print(
            f"{e['setting']}\t{e['predictor']}\t{e['horizon']:g}\t{g['n']}\t"
            f"{pp[base]['resume_ttft_p90']:.3f}\t"
            f"{pp[ours]['resume_ttft_p90']:.3f}\t"
            f"{g['mean']:+.1%}\t{g['median']:+.1%}\t"
            f"[{g['ci95'][0]:+.1%}, {g['ci95'][1]:+.1%}]\t{g['wins']}/{g['n']}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"args": vars(args), "rows": rows, "summary": summary}, f, indent=1
            )


if __name__ == "__main__":
    main()
