"""Go/no-go for restore timing: does a timed GPU warm-up beat warming up
immediately after each response?

Every warm-up variant sends exactly one extra request per turn (the same
hint-only request), so traffic is matched; only its send time differs:

    send_at = response finish + max(0, predicted gap - lead)

Baselines: `lru/none` (no warm-up) and `warm:inf/none` (warm-up at finish).
Leads and predictors are chosen on a tuning trace and reported on a held-out
trace sampled with a different seed.

Example:
    python -m kvplace.restore_sweep --tune traces/tl300_full_r05.jsonl \
        --test traces/tl300_full_r05_s1.jsonl --cpu-blocks 24000 96000 \
        --out results/restore_sweep.json

"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from kvplace.sim import CostModel, Simulator, get_policy
from kvplace.trace import read_trace

BASELINE = ("warm:inf", "none")
NO_WARMUP = ("lru", "none")
METRICS = (
    "resume_ttft_p50",
    "resume_ttft_p90",
    "resume_ttft_p99",
    "resume_frac_gpu",
    "hint_requests",
    "warmup_late",
    "warmup_cancelled",
)


def _run(job: tuple) -> dict:
    trace, policy, predictor, gpu_blocks, cpu_blocks = job
    sim = Simulator(
        read_trace(trace),
        get_policy(policy),
        predictor,
        CostModel(),
        gpu_blocks,
        cpu_blocks,
        10_000_000,
    )
    summary = sim.run().summary()
    return {
        "trace": trace,
        "cpu_blocks": cpu_blocks,
        **{k: summary[k] for k in ("policy", "predictor", *METRICS)},
    }


def _p90_gain(row: dict, base: dict) -> float:
    return 1 - row["resume_ttft_p90"] / base["resume_ttft_p90"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tune", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--gpu-blocks", type=int, default=12000)
    p.add_argument("--cpu-blocks", type=int, nargs="+", default=[24000, 96000])
    p.add_argument("--leads", type=float, nargs="+", default=[0.5, 2, 5, 15])
    p.add_argument("--predictors", nargs="+", default=["oracle", "tool", "toolq:0.25"])
    p.add_argument("--bar", type=float, default=0.15, help="required p90 gain")
    p.add_argument("--max-p50-regression", type=float, default=0.05)
    p.add_argument("--out")
    args = p.parse_args()

    arms = [NO_WARMUP, BASELINE] + [
        (f"warm:{lead:g}", pred) for pred in args.predictors for lead in args.leads
    ]
    jobs = [
        (trace, pol, pred, args.gpu_blocks, cpu)
        for trace in (args.tune, args.test)
        for cpu in args.cpu_blocks
        for pol, pred in arms
    ]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        rows = list(ex.map(_run, jobs))

    def get(trace, cpu, pol, pred):
        return next(
            r
            for r in rows
            if (r["trace"], r["cpu_blocks"], r["policy"], r["predictor"])
            == (trace, cpu, pol, pred)
        )

    print("\t".join(["split", "cpu_blocks", "policy", "predictor", *METRICS]))
    for r in rows:
        split = "tune" if r["trace"] == args.tune else "test"
        vals = [r[m] for m in METRICS]
        print(
            "\t".join(
                [split, str(r["cpu_blocks"]), r["policy"], r["predictor"]]
                + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals]
            )
        )

    decisions = []
    print("\nDecision (held-out p90 gain vs warm-up at finish):")
    for cpu in args.cpu_blocks:
        base = get(args.test, cpu, *BASELINE)
        for pred in args.predictors:
            # Choose the lead on the tuning trace only.
            lead_pol = max(
                (f"warm:{lead:g}" for lead in args.leads),
                key=lambda pol: _p90_gain(
                    get(args.tune, cpu, pol, pred), get(args.tune, cpu, *BASELINE)
                ),
            )
            row = get(args.test, cpu, lead_pol, pred)
            gain = _p90_gain(row, base)
            p50_change = row["resume_ttft_p50"] / base["resume_ttft_p50"] - 1
            ok = gain >= args.bar and p50_change <= args.max_p50_regression
            decisions.append(
                {
                    "cpu_blocks": cpu,
                    "predictor": pred,
                    "lead": lead_pol,
                    "p90_gain": gain,
                    "p50_change": p50_change,
                    "passes": ok,
                }
            )
            print(
                f"  cpu={cpu} {pred:<11} {lead_pol:<9} p90 {gain:+.1%} "
                f"p50 {p50_change:+.1%} -> {'PASS' if ok else 'fail'}"
            )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows, "decisions": decisions}, f)


if __name__ == "__main__":
    main()
