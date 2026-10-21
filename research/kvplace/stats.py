"""Summarize and compare traces (fit/validate the synthetic generator).

Example:
    python -m kvplace.stats traces/syn300.jsonl traces/tracelab_claude.jsonl

"""

import argparse
import math
import statistics
from collections import Counter

from kvplace.trace import read_trace


def _q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else float("nan")


def summarize(path: str) -> dict:
    sessions = read_trace(path)
    gaps, pairs, ctx, new_in, out = [], [], [], [], []
    tools: Counter[str] = Counter()
    for s in sessions:
        ctx.append(s.prompt_lengths()[-1])
        sg = []
        for t in s.turns:
            new_in.append(t.new_input_tokens)
            out.append(t.output_tokens)
            if t.tool_name is not None:
                gaps.append(t.tool_duration_s)
                tools[t.tool_name] += 1
                sg.append(math.log(t.tool_duration_s + 1e-3))
        pairs += list(zip(sg, sg[1:]))
    lag1 = statistics.correlation(*zip(*pairs)) if len(pairs) > 2 else float("nan")
    n_turns = [len(s.turns) for s in sessions]
    return {
        "sessions": len(sessions),
        "turns p50/p90": (_q(n_turns, 0.5), _q(n_turns, 0.9)),
        "gap s p50/p90/p99": tuple(round(_q(gaps, p), 2) for p in (0.5, 0.9, 0.99)),
        "gaps > 10s": round(sum(g > 10 for g in gaps) / max(1, len(gaps)), 3),
        "gaps > 60s": round(sum(g > 60 for g in gaps) / max(1, len(gaps)), 3),
        "lag-1 corr log(gap)": round(lag1, 3),
        "new input p50/p90": (_q(new_in, 0.5), _q(new_in, 0.9)),
        "output p50/p90": (_q(out, 0.5), _q(out, 0.9)),
        "final ctx p50/p90": (_q(ctx, 0.5), _q(ctx, 0.9)),
        "top tools": tools.most_common(6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("traces", nargs="+")
    args = p.parse_args()
    for path in args.traces:
        print(f"== {path}")
        for k, v in summarize(path).items():
            print(f"  {k:22} {v}")


if __name__ == "__main__":
    main()
