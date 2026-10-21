"""Generate a synthetic coding-agent trace.

Sessions arrive as a Poisson process. Each session runs a number of turns;
between turns the agent calls a tool whose duration is drawn from a per-tool
log-normal distribution. Tool mixes are rough stand-ins for coding agents
(fast file reads, slow test/build runs, occasional long human waits) and are
meant to be replaced with fitted distributions once real traces exist.

Example:
    python -m kvplace.gen_synthetic --sessions 200 --rate 0.5 -o trace.jsonl

"""

import argparse
import math
import random
from dataclasses import dataclass

from kvplace.trace import Session, Turn, write_trace


@dataclass(frozen=True)
class ToolSpec:
    weight: float
    median_s: float
    sigma: float  # log-space standard deviation
    result_tokens: tuple[int, int]  # uniform range of tool output tokens


DEFAULT_TOOLS: dict[str, ToolSpec] = {
    "read_file": ToolSpec(0.35, 0.3, 0.6, (200, 2000)),
    "grep": ToolSpec(0.20, 0.8, 0.7, (50, 800)),
    "edit_file": ToolSpec(0.20, 0.5, 0.5, (20, 200)),
    "run_tests": ToolSpec(0.15, 25.0, 0.9, (100, 3000)),
    "build": ToolSpec(0.07, 60.0, 0.8, (100, 1500)),
    "ask_human": ToolSpec(0.03, 120.0, 1.0, (20, 300)),
}


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return rng.lognormvariate(math.log(median), sigma)


def generate(
    num_sessions: int,
    arrival_rate: float,
    mean_turns: float,
    max_turns: int,
    task_tokens: tuple[int, int],
    output_tokens: tuple[int, int],
    shared_prefix_tokens: int,
    max_context_tokens: int,
    tools: dict[str, ToolSpec],
    seed: int,
) -> list[Session]:
    rng = random.Random(seed)
    names = list(tools)
    weights = [tools[n].weight for n in names]
    sessions = []
    t = 0.0
    for i in range(num_sessions):
        t += rng.expovariate(arrival_rate)
        n_turns = min(max_turns, 1 + int(rng.expovariate(1 / (mean_turns - 1))))
        turns: list[Turn] = []
        ctx = 0
        for k in range(n_turns):
            if k == 0:
                new_in = shared_prefix_tokens + rng.randint(*task_tokens)
            else:
                prev_tool = tools[turns[-1].tool_name]
                new_in = rng.randint(*prev_tool.result_tokens)
            out = rng.randint(*output_tokens)
            ctx += new_in
            last = k == n_turns - 1 or ctx + out + 4000 > max_context_tokens
            ctx += out
            if last:
                turns.append(Turn(new_in, out))
                break
            else:
                name = rng.choices(names, weights)[0]
                spec = tools[name]
                dur = _lognormal(rng, spec.median_s, spec.sigma)
                turns.append(Turn(new_in, out, name, round(dur, 3)))
        sessions.append(
            Session(
                session_id=f"s{i:05d}",
                start_s=round(t, 3),
                turns=turns,
                shared_prefix_tokens=shared_prefix_tokens,
                seed=rng.getrandbits(32),
            )
        )
    return sessions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--sessions", type=int, default=200)
    p.add_argument("--rate", type=float, default=0.5, help="sessions per second")
    p.add_argument("--mean-turns", type=float, default=12.0)
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--task-tokens", type=int, nargs=2, default=(1000, 4000))
    p.add_argument("--output-tokens", type=int, nargs=2, default=(50, 400))
    p.add_argument("--shared-prefix-tokens", type=int, default=2000)
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=30000,
        help="end a session before its context would exceed this",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sessions = generate(
        args.sessions,
        args.rate,
        args.mean_turns,
        args.max_turns,
        tuple(args.task_tokens),
        tuple(args.output_tokens),
        args.shared_prefix_tokens,
        args.max_context_tokens,
        DEFAULT_TOOLS,
        args.seed,
    )
    write_trace(sessions, args.output)
    n_turns = sum(len(s.turns) for s in sessions)
    max_ctx = max(max(s.prompt_lengths()) for s in sessions)
    print(f"wrote {len(sessions)} sessions, {n_turns} turns, max ctx {max_ctx}")


if __name__ == "__main__":
    main()
