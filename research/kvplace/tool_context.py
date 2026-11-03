"""How much context each tool contributes, and how much can be fetched again.

A turn's new input tokens are attributed to the tool that preceded it (the
slowest, if several ran). Tool inputs are redacted in the public traces, so this
cannot detect actual repeat reads; it bounds what dropping could lose for good.

    python -m kvplace.tool_context traces/fx/cat.jsonl
"""

import argparse
from collections import Counter

from kvplace.trace import read_trace

REFETCHABLE = {"Read", "Grep", "Glob", "LS", "WebFetch", "WebSearch", "NotebookRead"}
NOT_TOOLS = {"(start)", "human", "none"}


def contributions(path: str) -> tuple[Counter, int, int]:
    mass: Counter = Counter()
    sessions = read_trace(path)
    with_human = 0
    for s in sessions:
        human = False
        for k, turn in enumerate(s.turns):
            tool = s.turns[k - 1].tool_name if k else "(start)"
            mass[tool or "none"] += max(0, turn.new_input_tokens)
            human |= tool == "human"
        with_human += human
    return mass, with_human, len(sessions)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("traces", nargs="+")
    args = p.parse_args()
    for path in args.traces:
        mass, with_human, n = contributions(path)
        total = sum(mass.values())
        tools = total - sum(mass[k] for k in NOT_TOOLS)
        refetch = sum(v for k, v in mass.items() if k in REFETCHABLE)
        print(f"{path}: {with_human}/{n} sessions have a later human turn")
        print(f"  tool results are {tools / total:.2f} of added tokens")
        print(f"  refetchable tools: {refetch / tools:.2f} of tool tokens")
        for name, tokens in mass.most_common(9):
            print(f"  {name:20s} {tokens / total:.3f}")


if __name__ == "__main__":
    main()
