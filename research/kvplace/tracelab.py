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

Each segment's first-call provider cache hit (`prefix_tokens`) becomes its
`shared_prefix_tokens`, assuming that content (system prompt, tool
definitions) is common to all sessions; --no-shared-prefix disables this.

Each session's `meta` records where and why it ends (`end_reason`):
`session_end` (last row of the raw session), `compaction` (split at a context
shrink), `max_context` / `max_turns` (truncated). For the artificial ends it
also records what the trace does next (`end_next_tool`, labeled like a gap's
tool, and `end_next_gap_s`), which is what a live agent would know when its
last kept response finished (`tool-nfh2` in kvplace.predictors uses it). The
turns themselves are unchanged. `meta["user"]` is the TraceLab developer id.

--user-split train|test keeps only one side of a fixed developer-level split
(`user_side`: sha256 of "<split-seed>:<user>", test if below --test-frac);
--users-include / --users-exclude filter by explicit developer ids.

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
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime

from kvplace.trace import Session, Turn, write_trace

INPUT_EVENTS = {"tool_result", "user_message"}
END_REASONS = ("session_end", "compaction", "max_context", "max_turns")


def user_side(user: str | None, split_seed: int = 0, test_frac: float = 0.4) -> str:
    """Fixed developer-level split: "test" if the first 8 bytes of
    sha256("<seed>:<user>"), as a fraction of 2**64, are below `test_frac`,
    else "train". Stable across processes/platforms."""
    digest = hashlib.sha256(f"{split_seed}:{user}".encode()).digest()
    h = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if h < test_frac else "train"


def filter_users(
    rows_by_session: dict[str, list[dict]],
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    split: str | None = None,
    split_seed: int = 0,
    test_frac: float = 0.4,
) -> dict[str, list[dict]]:
    """Keep sessions whose developer (`user` of the session's rows) passes the
    include/exclude lists and, if `split` is set, falls on that side."""
    out = {}
    for sid, rows in rows_by_session.items():
        user = rows[0].get("user")
        if include is not None and user not in include:
            continue
        if exclude is not None and user in exclude:
            continue
        if split is not None and user_side(user, split_seed, test_frac) != split:
            continue
        out[sid] = rows
    return out


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
                    "hit": d.get("prefix_tokens") or 0,
                    "out": d["output_tokens"] or 0,
                    "start": max(inputs) if inputs else min(times),
                    "end": max(times),
                    "human": events[0]["event_type"] == "user_message",
                    "tool": max(tools, key=lambda x: x[0])[1] if tools else None,
                    "user": d.get("user"),
                }
            )
    return sessions


def to_sessions(
    rows_by_session: dict[str, list[dict]],
    token_scale: float,
    max_context: int,
    max_turns: int,
    min_turns: int,
    shared_prefix: bool = True,
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
            # Why this segment ends, and the raw row that would have followed.
            end_reason, nxt_row = "session_end", None
            if seg_idx + 1 < len(segments):
                end_reason, nxt_row = "compaction", segments[seg_idx + 1][0]
            for i, row in enumerate(seg):
                if i == 0:
                    new_in = row["in"]
                else:
                    new_in = row["in"] - seg[i - 1]["in"] - seg[i - 1]["out"]
                new_in = max(1, round(new_in * token_scale))
                out_tok = max(1, round(row["out"] * token_scale))
                if ctx + new_in + out_tok > max_context or len(turns) >= max_turns:
                    end_reason = (
                        "max_turns" if len(turns) >= max_turns else "max_context"
                    )
                    nxt_row = row
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
            # The provider's cache hit on a segment's first call is content
            # shared with earlier sessions (system prompt, tool definitions).
            shared = 0
            if shared_prefix:
                hit = round(seg[0].get("hit", 0) * token_scale)
                shared = min(hit, turns[0].new_input_tokens)
            meta = {"source": "tracelab", "end_reason": end_reason}
            if nxt_row is not None:
                last = seg[len(turns) - 1]
                meta["end_next_tool"] = (
                    "human" if nxt_row["human"] else (nxt_row["tool"] or "none")
                )
                meta["end_next_gap_s"] = round(
                    max(0.0, nxt_row["start"] - last["end"]), 3
                )
            if "user" in seg[0]:
                meta["user"] = seg[0]["user"]
            out.append(
                Session(
                    session_id=f"{sid}#{seg_idx}",
                    start_s=seg[0]["start"],
                    turns=turns,
                    shared_prefix_tokens=shared,
                    meta=meta,
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
        "--shared-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="model each segment's first-call provider cache hit as a prefix "
        "shared across sessions",
    )
    p.add_argument(
        "--rebase-rate",
        type=float,
        help="replace session starts with Poisson arrivals (sessions/s)",
    )
    p.add_argument("--sessions", type=int, help="keep this many (random sample)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--users-include", nargs="+", help="keep only these developers")
    p.add_argument("--users-exclude", nargs="+", help="drop these developers")
    p.add_argument(
        "--user-split",
        choices=["train", "test"],
        help="keep one side of the developer-level split (see user_side)",
    )
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.4)
    args = p.parse_args()

    rows = load_rows(args.input, args.provider)
    if args.users_include or args.users_exclude or args.user_split:
        rows = filter_users(
            rows,
            set(args.users_include) if args.users_include else None,
            set(args.users_exclude) if args.users_exclude else None,
            args.user_split,
            args.split_seed,
            args.test_frac,
        )
    sessions = to_sessions(
        rows,
        args.token_scale,
        args.max_context,
        args.max_turns,
        args.min_turns,
        args.shared_prefix,
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
