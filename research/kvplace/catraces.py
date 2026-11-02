"""Load CacheWise's CATraces into the row format of `kvplace.tracelab`.

Source: https://github.com/cachewise-project/cachewise-coding-traces, the
`parsed_traces/<project>/<session>/events.json` files (from its
extract_events.py). Convert with `kvplace.tracelab --format catraces`, which
applies the same session building, scaling and rebasing as for TraceLab.

Mapping per session:
  * Keep the session's most common model; other models are helper calls
    (titles, summaries) outside the agent loop.
  * One row per request_id: input = input + cache creation + cache read
    tokens, output and end from the request's last record.
  * Start: the latest result timestamp of the tools the previous request
    issued (the agent sends the next call when the tool returns). Records are
    stamped as responses arrive, so the request's first record would add the
    provider's time to first token (~5 s median) to every gap. After a user
    turn there is no such anchor; the first record is used, so those gaps run
    a few seconds long.
  * A row's tool is the slowest tool issued by the previous request; if that
    request issued none, the row starts with a user message (`human`).
  * `hit` (provider cache read) gives the shared prefix; `user` is the
    project id.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from kvplace.tracelab import _ts


def load_rows(parsed_dir: str) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for path in sorted(Path(parsed_dir).glob("*/*/events.json")):
        events = json.loads(path.read_text())
        llm = [e for e in events if e["event_type"] == "llm_call"]
        if not llm:
            continue
        model = Counter(e["model"] for e in llm).most_common(1)[0][0]
        requests: dict[str, list[dict]] = defaultdict(list)
        for e in llm:
            if e["model"] == model:
                requests[e["request_id"]].append(e)
        tools: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            if e["event_type"] == "tool_call":
                tools[e["associated_llm_call_id"]].append(e)

        project, session = path.parts[-3], path.parts[-2]
        rows, prev_tools, prev_done = [], None, None
        for k, recs in enumerate(requests.values()):
            last = recs[-1]
            issued = [t for r in recs for t in tools.get(r["message_id"], [])]
            slowest = max(
                issued, key=lambda t: t.get("execution_time_ms") or 0, default=None
            )
            rows.append(
                {
                    "k": k,
                    "in": last["input_tokens"]
                    + last["cache_creation_input_tokens"]
                    + last["cache_read_input_tokens"],
                    "hit": last["cache_read_input_tokens"],
                    "out": last["output_tokens"] or 0,
                    "start": prev_done or _ts(recs[0]["timestamp"]),
                    "end": max(_ts(r["timestamp"]) for r in recs),
                    "human": k > 0 and prev_tools is None,
                    "tool": prev_tools,
                    "user": project,
                }
            )
            prev_tools = slowest["tool_name"] if slowest else None
            done = [
                _ts(t["result_timestamp"]) for t in issued if t.get("result_timestamp")
            ]
            prev_done = max(done) if done else None
        sessions[f"{project}/{session}"] = rows
    return sessions
