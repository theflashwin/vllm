"""Live LRU-vs-ours replay on a Modal L4: modal_live.py --trace <jsonl>.

    uvx --from modal modal run research/modal_live.py \
        --trace research/traces/live/smoke20.jsonl
    uvx --from modal modal run research/modal_live.py \
        --trace research/traces/live/paired100.jsonl --run paired100_a

For each config, starts `vllm serve` with the OffloadingConnector (CPU tier
plus an fs disk tier on the container's local disk), replays the trace with
`kvplace.replay`, and saves requests.csv, metrics_delta.json, server.log and
summary.txt under research/results/live/<run>/<config>/. Configs mirror
scripts/run_baselines.sh; tier_lru vs tier_reuse is the paired comparison.
Model weights are cached in the `kvplace-hf-cache` volume. Set KVPLACE_GPU
(default L4) to run on another GPU type.
"""

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU = os.environ.get("KVPLACE_GPU", "L4")
VENV = "/opt/kvplace/.venv"
PY = f"{VENV}/bin/python"
FS_DIR = "/tmp/kvplace_fs"
PORT = 8000

app = modal.App("kvplace-live")
hf_cache = modal.Volume.from_name("kvplace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.11.8")
    .run_commands(
        f"uv venv {VENV} --python 3.12",
        f"uv pip install --python {PY} vllm==0.31.0 aiohttp",
    )
    .env(
        {
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PYTHONPATH": "/experiment",
            "HF_HOME": "/hf",
            "KVPLACE_LOG_HINTS": "1",
        }
    )
    .add_local_dir(Path(__file__).parent / "kvplace", "/experiment/kvplace")
)


def kv_config(cfg: str, block_size: int, cpu_bytes: int) -> str | None:
    if cfg == "gpu_only":
        return None
    tiers, policy = cfg.split("_")
    extra: dict = {"block_size": block_size, "cpu_bytes_to_use": cpu_bytes}
    if policy in ("lru", "arc"):
        extra["eviction_policy"] = policy
    elif policy == "reuse":
        extra["eviction_policy"] = "ReuseAwareCachePolicy"
        extra["cache_policy_module_path"] = "kvplace.vllm_policy"
    else:
        raise ValueError(f"unknown config {cfg!r}")
    if tiers == "tier":
        extra["spec_name"] = "TieringOffloadingSpec"
        extra["secondary_tiers"] = [{"type": "fs", "root_dir": FS_DIR}]
    elif tiers != "cpu":
        raise ValueError(f"unknown config {cfg!r}")
    return json.dumps(
        {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
    )


def wait_healthy(server: subprocess.Popen, timeout_s: float = 1800) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError("vLLM server exited during startup")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM server did not become healthy")


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=49152,
    volumes={"/hf": hf_cache},
    timeout=6 * 3600,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def run_live(
    trace: str,
    configs: list[str],
    predictor: str,
    time_scale: float,
    gpu_blocks: int,
    cpu_bytes: int,
    max_len: int,
    block_size: int,
) -> dict[str, dict[str, bytes]]:
    trace_path = Path("/tmp/trace.jsonl")
    trace_path.write_text(trace)
    results = {}
    for cfg in configs:
        out = Path("/tmp/out") / cfg
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rm", "-rf", FS_DIR], check=True)
        Path(FS_DIR).mkdir(parents=True)
        cmd = [
            f"{VENV}/bin/vllm", "serve", MODEL, "--port", str(PORT),
            "--block-size", str(block_size), "--max-model-len", str(max_len),
            "--num-gpu-blocks-override", str(gpu_blocks),
            "--enable-prompt-tokens-details",
        ]  # fmt: skip
        if (kv := kv_config(cfg, block_size, cpu_bytes)) is not None:
            cmd += ["--kv-transfer-config", kv]
        with open(out / "server.log", "w") as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_healthy(server)
                hf_cache.commit()
                replay = subprocess.run(
                    [PY, "-m", "kvplace.replay", str(trace_path),
                     "--url", f"http://localhost:{PORT}",
                     "--predictor", predictor, "--time-scale", str(time_scale),
                     "--block-size", str(block_size), "--out", str(out)],
                    capture_output=True, text=True,
                )  # fmt: skip
                (out / "summary.txt").write_text(replay.stdout + replay.stderr)
            finally:
                server.terminate()
                server.wait(timeout=120)
        results[cfg] = {p.name: p.read_bytes() for p in out.iterdir()}
    return results


@app.local_entrypoint()
def main(
    trace: str,
    configs: str = "tier_lru tier_reuse",
    predictor: str = "tool-nfh2",
    time_scale: float = 0.2,
    gpu_blocks: int = 3000,
    cpu_gb: float = 10.0,
    max_len: int = 32768,
    block_size: int = 16,
    run: str = "",
):
    run = run or f"{Path(trace).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    root = Path(__file__).parent / "results" / "live" / run
    args = dict(
        configs=configs.split(),
        predictor=predictor,
        time_scale=time_scale,
        gpu_blocks=gpu_blocks,
        cpu_bytes=int(cpu_gb * 1e9),
        max_len=max_len,
        block_size=block_size,
    )
    results = run_live.remote(Path(trace).read_text(), **args)
    root.mkdir(parents=True, exist_ok=True)
    meta = {"trace": trace, "gpu": GPU, **args}
    (root / "args.json").write_text(json.dumps(meta, indent=1))
    for cfg, files in results.items():
        (root / cfg).mkdir(exist_ok=True)
        for name, data in files.items():
            (root / cfg / name).write_bytes(data)
        hints = files.get("server.log", b"").count(b"kvplace hint")
        summary = files.get("summary.txt", b"").decode().strip()
        print(f"{cfg}: {summary} | policy saw {hints} hints")
    print(f"saved {root}")
