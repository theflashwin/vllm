"""Measure CostModel parameters on a Modal GPU (one bounded run per GPU type).

    uvx --from modal modal run research/modal_microbench.py --gpu L4
    uvx --from modal modal run research/modal_microbench.py --gpu H100

Runs `kvplace.microbench` gpu (pinned H2D/D2H), fs (the container's local
disk, O_DIRECT reads) and server (prefill rate and decode time from a
`vllm serve` instance) and saves the merged costs JSON locally.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PY = "/opt/kvplace/.venv/bin/python"

app = modal.App("kvplace-microbench")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.11.8")
    .run_commands(
        "uv venv /opt/kvplace/.venv --python 3.12",
        f"uv pip install --python {PY} vllm==0.30.0 requests",
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTHONPATH": "/experiment"})
    .add_local_dir(Path(__file__).parent / "kvplace", "/experiment/kvplace")
)


def _bench(gpu: str) -> dict:
    out = "/tmp/costs.json"
    bench = [PY, "-m", "kvplace.microbench"]
    subprocess.run([*bench, "gpu", "--model", MODEL, "--out", out], check=True)
    subprocess.run(
        [*bench, "fs", "--model", MODEL, "--dir", "/tmp/kvplace_fs", "--out", out],
        check=True,
    )
    server = subprocess.Popen(
        [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
         "--block-size", "16", "--max-model-len", "20000", "--port", "8000"]
    )  # fmt: skip
    try:
        import urllib.request

        for _ in range(180):
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("vLLM server did not become healthy")
        subprocess.run([*bench, "server", "--out", out], check=True)
    finally:
        server.terminate()
        server.wait(timeout=60)
    costs = json.loads(Path(out).read_text())
    costs["_gpu_type"] = gpu
    costs["_model"] = MODEL
    return costs


common = dict(
    image=image,
    cpu=(4, 8),
    memory=(32768, 49152),
    timeout=1500,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)


@app.function(gpu="L4", **common)
def bench_l4() -> dict:
    return _bench("L4")


@app.function(gpu="H100", **common)
def bench_h100() -> dict:
    return _bench("H100")


@app.local_entrypoint()
def main(gpu: str = "L4", out: str = ""):
    fn = {"L4": bench_l4, "H100": bench_h100}[gpu]
    costs = fn.remote()
    path = Path(out or Path(__file__).parent / f"costs_{gpu.lower()}.json")
    path.write_text(json.dumps(costs, indent=1))
    print({k: v for k, v in costs.items() if not k.startswith("_")})
    print(f"saved {path}")
