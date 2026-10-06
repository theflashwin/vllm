"""Measure the transfer and compute costs that parameterize the cost model.

Subcommands (each writes/merges into one JSON matching `kvplace.sim.CostModel`):

  gpu     GPU<->pinned-CPU copies of N offload chunks, both batched (one
          contiguous copy) and per-chunk (one copy per chunk, like a
          non-coalesced swap). Fits latency + bandwidth.
  fs      Write/read chunk-sized files on the secondary-tier filesystem with a
          thread pool; reads use O_DIRECT on Linux to bypass the page cache.
  server  Prefill throughput and per-token decode time from a running vLLM
          server (prefix-cache-miss prompts of increasing length).

Example (on the GPU box):
    python -m kvplace.microbench gpu --out costs.json
    python -m kvplace.microbench fs --dir /mnt/nvme/kvtest --out costs.json
    python -m kvplace.microbench server --url http://localhost:8000 --out costs.json
"""

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a + b*x; returns (a, b)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return my - b * mx, b


def kv_bytes_per_token(model: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model)
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2  # bf16


def bench_gpu(args) -> dict:
    import torch

    chunk = args.block_tokens * args.kv_bytes_per_token
    max_n = max(args.chunks)
    gpu = torch.empty(max_n * chunk, dtype=torch.uint8, device="cuda")
    cpu = torch.empty(max_n * chunk, dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream()

    def timed(fn) -> float:
        times = []
        for i in range(args.iters + 2):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record()
                fn()
                end.record()
            end.synchronize()
            if i >= 2:
                times.append(start.elapsed_time(end) / 1e3)
        return statistics.median(times)

    rows = []
    for n in args.chunks:
        nbytes = n * chunk
        perm = random.Random(n).sample(range(max_n), n)
        for direction in ("h2d", "d2h"):
            src, dst = (cpu, gpu) if direction == "h2d" else (gpu, cpu)

            def batched_copy(src=src, dst=dst, nbytes=nbytes):
                dst[:nbytes].copy_(src[:nbytes], non_blocking=True)

            def per_chunk(src=src, dst=dst, perm=perm):
                for j, p in enumerate(perm):
                    dst[j * chunk : (j + 1) * chunk].copy_(
                        src[p * chunk : (p + 1) * chunk], non_blocking=True
                    )

            batched = timed(batched_copy)
            scattered = timed(per_chunk)
            rows.append(
                {
                    "n": n,
                    "dir": direction,
                    "batched_s": batched,
                    "per_chunk_s": scattered,
                }
            )
            print(rows[-1])

    h2d = [r for r in rows if r["dir"] == "h2d"]
    a, b = fit_linear([r["n"] for r in h2d], [r["batched_s"] for r in h2d])
    return {
        "block_tokens": args.block_tokens,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "gpu_cpu_lat_s": max(a, 0.0),
        "gpu_cpu_gbps": chunk / b / 1e9,
        "_gpu_rows": rows,
    }


def bench_fs(args) -> dict:
    chunk = args.block_tokens * args.kv_bytes_per_token
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(chunk)
    direct = getattr(os, "O_DIRECT", 0)
    if not direct:
        print("warning: no O_DIRECT; reads may hit the page cache")

    def write(i: int) -> None:
        fd = os.open(root / f"{i}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)

    def read(i: int) -> None:
        import mmap

        fd = os.open(root / f"{i}.bin", os.O_RDONLY | direct)
        try:
            buf = mmap.mmap(-1, chunk)  # page-aligned, as O_DIRECT requires
            os.readv(fd, [buf])
            buf.close()
        finally:
            os.close(fd)

    rows = []
    with ThreadPoolExecutor(args.threads) as pool:
        for n in args.chunks:
            for op, fn in (("write", write), ("read", read)):
                times = []
                for _ in range(args.iters):
                    t = time.perf_counter()
                    list(pool.map(fn, range(n)))
                    times.append(time.perf_counter() - t)
                rows.append({"n": n, "op": op, "s": statistics.median(times)})
                print(rows[-1])
    for f in root.glob("*.bin"):
        f.unlink()

    reads = [r for r in rows if r["op"] == "read"]
    a, b = fit_linear([r["n"] for r in reads], [r["s"] for r in reads])
    return {
        "sec_cpu_lat_s": max(a, 0.0),
        "sec_cpu_gbps": chunk / b / 1e9,
        "_fs_rows": rows,
    }


def bench_server(args) -> dict:
    import requests

    model = requests.get(f"{args.url}/v1/models").json()["data"][0]["id"]
    rng = random.Random(1234)

    def ttft(n_tokens: int, max_tokens: int) -> tuple[float, float]:
        prompt = [rng.randrange(1000, 20000) for _ in range(n_tokens)]
        t = time.perf_counter()
        first = None
        with requests.post(
            f"{args.url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": True,
            },
            stream=True,
        ) as r:
            for line in r.iter_lines():
                if line.startswith(b"data:") and first is None:
                    first = time.perf_counter() - t
        return first, time.perf_counter() - t

    ttft(256, 1)  # warmup
    xs, ys = [], []
    for n in args.prompt_lens:
        for _ in range(args.iters):
            xs.append(n)
            ys.append(ttft(n, 1)[0])
    a, b = fit_linear(xs, ys)
    first, total = ttft(512, 257)
    return {
        "sched_overhead_s": max(a, 0.0),
        "prefill_tok_per_s": 1 / b,
        "decode_s_per_tok": (total - first) / 256,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("gpu", "fs", "server"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--iters", type=int, default=5)
        s.add_argument("--block-tokens", type=int, default=16)
        s.add_argument("--kv-bytes-per-token", type=int)
        s.add_argument(
            "--model",
            default="Qwen/Qwen2.5-7B-Instruct",
            help="used to derive --kv-bytes-per-token",
        )
        s.add_argument(
            "--chunks", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024]
        )
    sub.choices["fs"].add_argument("--dir", required=True)
    sub.choices["fs"].add_argument("--threads", type=int, default=16)
    sub.choices["server"].add_argument("--url", default="http://localhost:8000")
    sub.choices["server"].add_argument(
        "--prompt-lens", type=int, nargs="+", default=[512, 2048, 4096, 8192, 16384]
    )
    args = p.parse_args()
    if args.cmd != "server" and args.kv_bytes_per_token is None:
        args.kv_bytes_per_token = kv_bytes_per_token(args.model)

    result = {"gpu": bench_gpu, "fs": bench_fs, "server": bench_server}[args.cmd](args)
    out = Path(args.out)
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(result)
    out.write_text(json.dumps(merged, indent=1))
    print({k: v for k, v in result.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
