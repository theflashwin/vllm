"""Convert the TraceLab coding-agent trace into kvplace's trace format.

Source: https://github.com/uw-syfi/TraceLab (CC BY 4.0), file
`syfi_coding_trace.jsonl.gz`: one JSON row per LLM call ("round") of Claude
Code / Codex sessions.

Mapping (row k of a session, sorted by round_index):
  * request start = latest input event (tool_result / user_message), else the
    earliest event; response end = latest event of the row.
  * new_input_tokens = input_tokens_total[k] - input_tokens_total[k-1]
    - output_tokens[k-1]. `prefix_tokens` is the provider's cache hit, not the
    logical prefix, so it is not used. A negative value means the client
    compacted or rewrote the context; the session is split there and the new
    segment starts with its whole context as fresh input.
  * gap after turn k = start[k+1] - end[k] (clamped at 0). Its "tool" is
    `human` if row k+1 begins with a user message, else the slowest tool whose
    result feeds row k+1. For Claude rows this gap matches the recorded
    `tool_wall_latency_ms`; Codex timestamps are less consistent (see README).

Real contexts (30k-200k tokens) exceed small models' windows, so
--token-scale shrinks every token count and --max-context truncates sessions.
The trace spans weeks; --rebase-rate replaces session start times with a
Poisson process while keeping each session's internal timing.

Example:
    python -m kvplace.tracelab traces/syfi_coding_trace.jsonl.gz \
        -o traces/tracelab_claude.jsonl --provider claude --token-scale 0.25 \
        --max-context 30000 --rebase-rate 0.3 --sessions 300

"""

import argparse
import gzip
import json
import random
from collections import defaultdict
from datetime import datetime

from kvplace.trace import Session, Turn, write_trace

INPUT_EVENTS = {"tool_result", "user_message"}


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_rows(path: str, provider: str) -> dict[str, list[dict]]:
    """Keep only the fields we need, grouped by session."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if provider != "all" and d["provider"] != provider:
                continue
            events = d["timing_events"]
            if not events:
                continue
            times = [_ts(e["timestamp"]) for e in events]
            inputs = [
                t for e, t in zip(events, times) if e["event_type"] in INPUT_EVENTS
            ]
            tools = [
                (t["tool_wall_latency_ms"] or 0, t["tool_name"]) for t in d["tools"]
            ]
            sessions[d["session_id"]].append(
                {
                    "k": d["round_index"],
                    "in": d["input_tokens_total"],
                    "out": d["output_tokens"] or 0,
                    "start": max(inputs) if inputs else min(times),
                    "end": max(times),
                    "human": events[0]["event_type"] == "user_message",
                    "tool": max(tools, key=lambda x: x[0])[1] if tools else None,
                }
            )
    return sessions


def to_sessions(
    rows_by_session: dict[str, list[dict]],
    token_scale: float,
    max_context: int,
    max_turns: int,
    min_turns: int,
) -> list[Session]:
    out = []
    for sid, rows in rows_by_session.items():
        rows.sort(key=lambda r: r["k"])
        segments: list[list[dict]] = [[]]
        for prev, row in zip([None, *rows], rows):
            if prev is not None and row["in"] < prev["in"] + prev["out"]:
                segments.append([])  # context compacted / rewritten
            segments[-1].append(row)
        for seg_idx, seg in enumerate(segments):
            turns: list[Turn] = []
            ctx = 0
            for i, row in enumerate(seg):
                if i == 0:
                    new_in = row["in"]
                else:
                    new_in = row["in"] - seg[i - 1]["in"] - seg[i - 1]["out"]
                new_in = max(1, round(new_in * token_scale))
                out_tok = max(1, round(row["out"] * token_scale))
                if ctx + new_in + out_tok > max_context or len(turns) >= max_turns:
                    break
                ctx += new_in + out_tok
                if i + 1 < len(seg):
                    nxt = seg[i + 1]
                    tool = "human" if nxt["human"] else (nxt["tool"] or "none")
                    gap = max(0.0, nxt["start"] - row["end"])
                    turns.append(Turn(new_in, out_tok, tool, round(gap, 3)))
                else:
                    turns.append(Turn(new_in, out_tok))
            if len(turns) < min_turns:
                continue
            # A truncated session ends on its last kept turn.
            turns[-1] = Turn(turns[-1].new_input_tokens, turns[-1].output_tokens)
            out.append(
                Session(
                    session_id=f"{sid}#{seg_idx}",
                    start_s=seg[0]["start"],
                    turns=turns,
                    meta={"source": "tracelab"},
                )
            )
    out.sort(key=lambda s: s.start_s)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", help="syfi_coding_trace.jsonl.gz")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--provider", choices=["claude", "codex", "all"], default="claude")
    p.add_argument("--token-scale", type=float, default=1.0)
    p.add_argument("--max-context", type=int, default=10**9)
    p.add_argument("--max-turns", type=int, default=10**9)
    p.add_argument("--min-turns", type=int, default=2)
    p.add_argument(
        "--rebase-rate",
        type=float,
        help="replace session starts with Poisson arrivals (sessions/s)",
    )
    p.add_argument("--sessions", type=int, help="keep this many (random sample)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sessions = to_sessions(
        load_rows(args.input, args.provider),
        args.token_scale,
        args.max_context,
        args.max_turns,
        args.min_turns,
    )
    rng = random.Random(args.seed)
    if args.sessions and args.sessions < len(sessions):
        sessions = sorted(rng.sample(sessions, args.sessions), key=lambda s: s.start_s)
    if args.rebase_rate:
        t = 0.0
        for s in sessions:
            t += rng.expovariate(args.rebase_rate)
            s.start_s = round(t, 3)
    else:
        t0 = sessions[0].start_s
        for s in sessions:
            s.start_s = round(s.start_s - t0, 3)
    for i, s in enumerate(sessions):
        s.seed = i
    write_trace(sessions, args.output)
    n_turns = sum(len(s.turns) for s in sessions)
    print(f"wrote {len(sessions)} sessions, {n_turns} turns to {args.output}")


if __name__ == "__main__":
    main()
