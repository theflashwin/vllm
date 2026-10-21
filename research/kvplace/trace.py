"""Agentic workload trace schema.

A trace is a JSONL file with one `Session` per line. Each session is a
sequence of turns; turn k+1's prompt is turn k's prompt plus the assistant
output and the tool result, so consecutive turns share a growing prefix.
After each non-final turn the agent spends `tool_duration_s` executing a tool
before submitting the next turn — that gap is the reuse horizon the placement
policy tries to predict.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Turn:
    # Tokens appended to the context before this turn's request (system prompt
    # + task for turn 0; previous assistant output + tool result afterwards).
    new_input_tokens: int
    output_tokens: int
    # Tool executed after this turn completes; None on the final turn.
    tool_name: str | None = None
    tool_duration_s: float = 0.0


@dataclass
class Session:
    session_id: str
    start_s: float
    turns: list[Turn]
    # Tokens shared with every other session (e.g. a common system prompt).
    shared_prefix_tokens: int = 0
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def prompt_lengths(self) -> list[int]:
        """Prompt length (tokens) of each turn's request."""
        lengths = []
        total = 0
        for i, turn in enumerate(self.turns):
            if i > 0:
                total += self.turns[i - 1].output_tokens
            total += turn.new_input_tokens
            lengths.append(total)
        return lengths


def write_trace(sessions: Iterable[Session], path: str | Path) -> None:
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(asdict(s)) + "\n")


def read_trace(path: str | Path) -> list[Session]:
    return list(iter_trace(path))


def iter_trace(path: str | Path) -> Iterator[Session]:
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["turns"] = [Turn(**t) for t in d["turns"]]
            yield Session(**d)
