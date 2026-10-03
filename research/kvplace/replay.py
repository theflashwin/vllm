"""Replay an agentic trace against a running vLLM OpenAI server.

Each session runs as a coroutine: send turn k, stream the response, sleep for
the tool duration, send turn k+1. Prompts are token-ID lists so consecutive
turns share an exact prefix (no chat-template or tokenizer drift). Turn k+1's
prompt contains synthetic stand-ins for turn k's output, so those tokens are
recomputed — the simulator models them the same way.

With a post-response predictor (e.g. `tool`), each turn's request carries no
hint; instead a hint-only request (same prefix + one padding block,
max_tokens=1) is sent right after the response, once the agent knows its tool.

Writes one CSV row per request plus a JSON delta of the server's Prometheus
counters (offload bytes/time, prefix-cache hits) over the run.

Example:
    python -m kvplace.replay trace.jsonl --url http://localhost:8000 \
        --predictor oracle --out results/cpu_lru

"""

import argparse
import asyncio
import csv
import json
import random
import re
import time
import zlib
from pathlib import Path

import aiohttp

from kvplace.hints import ReuseHint
from kvplace.predictors import get_predictor, is_post_response
from kvplace.trace import Session, read_trace

VOCAB_RANGE = (1000, 20000)  # avoid special tokens in common tokenizers
METRIC_RE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$")
METRIC_PREFIXES = (
    "vllm:kv_offload",
    "vllm:prefix_cache",
    "vllm:external_prefix_cache",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
)


def session_tokens(session: Session, shared: list[int]) -> list[int]:
    total = session.prompt_lengths()[-1]
    rng = random.Random(session.seed)
    private = [rng.randrange(*VOCAB_RANGE) for _ in range(total)]
    n_shared = min(session.shared_prefix_tokens, total)
    return shared[:n_shared] + private[n_shared:]


async def scrape_metrics(http: aiohttp.ClientSession, url: str) -> dict[str, float]:
    async with http.get(f"{url}/metrics") as r:
        text = await r.text()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if m and m.group(1).startswith(METRIC_PREFIXES):
            key = m.group(1) + (m.group(2) or "")
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def scale_hint(hint: ReuseHint | None, time_scale: float) -> ReuseHint | None:
    if hint is None or hint.expected_reuse_s is None:
        return hint
    return ReuseHint(
        hint.session_id, hint.turn, hint.expected_reuse_s * time_scale, hint.final
    )


def hint_padding(session_id: str, k: int, prompt_len: int, block: int) -> list[int]:
    """Fresh tokens that complete exactly one new offload block after the
    prompt. Storing that block is what delivers the hint: in tiered mode the
    CPU tier only tracks requests that store or load a chunk, and the
    hint-only request's prefix is usually all GPU hits."""
    rng = random.Random(zlib.crc32(f"{session_id}:{k}:pad".encode()))
    n = block - prompt_len % block
    return [rng.randrange(*VOCAB_RANGE) for _ in range(n)]


async def send_hint_request(http, args, model, prompt, hint, row) -> None:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "kv_transfer_params": hint.to_kv_transfer_params(),
    }
    sent = time.time()
    async with http.post(f"{args.url}/v1/completions", json=body) as r:
        r.raise_for_status()
        await r.read()
    row["e2e_s"] = time.time() - sent


async def run_session(http, args, model, session, shared, predictor, t0, rows):
    await asyncio.sleep(max(0.0, t0 + session.start_s * args.time_scale - time.time()))
    tokens = session_tokens(session, shared)
    lengths = session.prompt_lengths()
    post = is_post_response(predictor)
    hint_tasks = []
    gaps: list[float] = []  # trace time; predictors never see time_scale
    for k, turn in enumerate(session.turns):
        hint = None if post else predictor(session, k, gaps)
        hint = scale_hint(hint, args.time_scale)
        body = {
            "model": model,
            "prompt": tokens[: lengths[k]],
            "max_tokens": turn.output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if hint is not None:
            body["kv_transfer_params"] = hint.to_kv_transfer_params()
        sent = time.time()
        ttft = None
        usage = {}
        async with http.post(f"{args.url}/v1/completions", json=body) as r:
            r.raise_for_status()
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if ttft is None and chunk.get("choices"):
                    ttft = time.time() - sent
                if chunk.get("usage"):
                    usage = chunk["usage"]
        done = time.time()
        details = usage.get("prompt_tokens_details") or {}
        row = {
            "kind": "turn",
            "session_id": session.session_id,
            "turn": k,
            "sent_s": sent - t0,
            "ttft_s": ttft,
            "e2e_s": done - sent,
            "prompt_tokens": lengths[k],
            "cached_tokens": details.get("cached_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "gap_before_s": gaps[-1] * args.time_scale if gaps else 0.0,
            "hint_reuse_s": hint.expected_reuse_s if hint else None,
            "hint_final": hint.final if hint else None,
        }
        rows.append(row)

        if post:
            # The agent now knows its tool call (or that it is done).
            post_hint = scale_hint(predictor(session, k, gaps), args.time_scale)
            if post_hint is not None:
                prompt = tokens[: lengths[k]] + hint_padding(
                    session.session_id, k, lengths[k], args.block_size
                )
                hint_row = {
                    **row,
                    "kind": "hint",
                    "sent_s": time.time() - t0,
                    "ttft_s": None,
                    "prompt_tokens": len(prompt),
                    "cached_tokens": None,
                    "output_tokens": 1,
                    "hint_reuse_s": post_hint.expected_reuse_s,
                    "hint_final": post_hint.final,
                }
                rows.append(hint_row)
                # Runs concurrently with the tool; never delays the next turn.
                hint_tasks.append(
                    asyncio.create_task(
                        send_hint_request(
                            http, args, model, prompt, post_hint, hint_row
                        )
                    )
                )

        if k + 1 < len(session.turns):
            await asyncio.sleep(turn.tool_duration_s * args.time_scale)
            gaps.append(turn.tool_duration_s)
            if hasattr(predictor, "observe"):
                predictor.observe(turn.tool_name, turn.tool_duration_s)
    await asyncio.gather(*hint_tasks)


async def main_async(args) -> None:
    sessions = read_trace(args.trace)
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    predictor = get_predictor(args.predictor)
    rng = random.Random(0)
    max_shared = max(s.shared_prefix_tokens for s in sessions)
    shared = [rng.randrange(*VOCAB_RANGE) for _ in range(max_shared)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        async with http.get(f"{args.url}/v1/models") as r:
            model = args.model or (await r.json())["data"][0]["id"]
        before = await scrape_metrics(http, args.url)
        rows: list[dict] = []
        t0 = time.time()
        await asyncio.gather(
            *(
                run_session(http, args, model, s, shared, predictor, t0, rows)
                for s in sessions
            )
        )
        wall = time.time() - t0
        after = await scrape_metrics(http, args.url)

    rows.sort(key=lambda r: r["sent_s"])
    with open(out / "requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    with open(out / "metrics_delta.json", "w") as f:
        json.dump({"wall_s": wall, "args": vars(args), "metrics": delta}, f, indent=1)
    resume = sorted(
        r["ttft_s"]
        for r in rows
        if r["kind"] == "turn" and r["turn"] > 0 and r["ttft_s"]
    )
    if resume:
        p = lambda q: resume[int(q * (len(resume) - 1))]  # noqa: E731
        print(
            f"{len(rows)} requests in {wall:.0f}s; resume TTFT "
            f"p50={p(0.5):.3f} p90={p(0.9):.3f} p99={p(0.99):.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", help="defaults to the server's first model")
    p.add_argument("--predictor", default="none")
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="multiply session start times and tool durations",
    )
    p.add_argument("--max-sessions", type=int)
    p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="offload block_size (tokens); hint-only requests pad to it",
    )
    p.add_argument("--out", required=True, help="output directory")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
