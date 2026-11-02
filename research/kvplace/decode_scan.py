"""Measure decode step time against context length and concurrency.

Tests the scan model: each decode step reads every running request's KV, so
step = a + b * (total context tokens), and n requests of L tokens cost the same
as one request of n*L tokens. Also measures incremental prefill: fresh tokens
appended to a cached prefix of L tokens.

Each cell warms n random prompts into the GPU prefix cache one at a time, then
sends them together so all n decode in lockstep with almost no prefill. Cells
whose prompts did not hit the cache are flagged by `hit` in the output.

    python -m kvplace.decode_scan --url http://localhost:8000 --model M \
        --max-total 120000 --out scan.jsonl
"""

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import time

import aiohttp

SUFFIX_TOKENS = 512


def gpu_utilization() -> tuple[float, float] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout  # fmt: skip
        core, mem = out.strip().split(", ")
        return float(core), float(mem)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def random_prompt(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(1000, 30000) for _ in range(n)]


async def complete(
    http: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
) -> dict:
    """Stream one completion; returns TTFT, step time and cached tokens."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = last = None
    usage: dict = {}
    async with http.post(f"{url}/v1/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            if chunk.get("choices"):
                last = time.perf_counter()
                first = first or last
            usage = chunk.get("usage") or usage
    out = usage["completion_tokens"]
    return {
        "ttft_s": first - start,
        "step_s": (last - first) / (out - 1) if out > 1 else None,
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "prompt": usage["prompt_tokens"],
    }


async def run_cell(
    http: aiohttp.ClientSession,
    url: str,
    model: str,
    ctx: int,
    n: int,
    max_tokens: int,
    rng: random.Random,
) -> dict:
    prompts = [random_prompt(ctx, rng) for _ in range(n)]
    warm = [await complete(http, url, model, p, 1) for p in prompts]

    util: list[tuple[float, float]] = []

    async def sample() -> None:
        while True:
            await asyncio.sleep(1.0)
            if (u := await asyncio.to_thread(gpu_utilization)) is not None:
                util.append(u)

    sampler = asyncio.create_task(sample())
    runs = await asyncio.gather(
        *(complete(http, url, model, p, max_tokens) for p in prompts)
    )
    sampler.cancel()

    suffix = await complete(
        http, url, model, prompts[0] + random_prompt(SUFFIX_TOKENS, rng), 1
    )
    return {
        "ctx": ctx,
        "n": n,
        "total": ctx * n,
        "step_s": statistics.median(r["step_s"] for r in runs),
        "step_max_s": max(r["step_s"] for r in runs),
        "ttft_s": statistics.median(r["ttft_s"] for r in runs),
        "hit": sum(r["cached"] for r in runs) / sum(r["prompt"] for r in runs),
        "cold_prefill_tok_s": ctx / statistics.median(w["ttft_s"] for w in warm),
        "suffix_tok_s": SUFFIX_TOKENS / suffix["ttft_s"],
        "suffix_hit": suffix["cached"] / suffix["prompt"],
        "gpu_util": statistics.fmean(u[0] for u in util) if util else None,
        "mem_util": statistics.fmean(u[1] for u in util) if util else None,
    }


async def main_async(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=None)
    cells = [
        (ctx, n)
        for ctx in args.contexts
        for n in args.concurrency
        if ctx * n <= args.max_total
    ]
    async with aiohttp.ClientSession(timeout=timeout) as http:
        for rep in range(args.repeats):
            for ctx, n in cells:
                try:
                    row = await run_cell(
                        http, args.url, args.model, ctx, n, args.max_tokens, rng
                    )
                except Exception as e:  # keep the remaining cells on failure
                    row = {"ctx": ctx, "n": n, "error": repr(e)}
                row["rep"] = rep
                print(json.dumps(row), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--contexts", type=int, nargs="+", default=[2000, 8000, 32000, 80000]
    )
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--max-total", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
