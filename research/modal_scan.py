"""Run `kvplace.decode_scan` against vLLM on a Modal GPU.

    KVPLACE_GPU=A10G uvx --from modal modal run --detach \
        research/modal_scan.py --run scan_a10g --max-total 120000
    modal volume get kvplace-results scan_a10g

Rows are saved to the `kvplace-results` volume after every cell.
"""

import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU = os.environ.get("KVPLACE_GPU", "A10G")
VENV = "/opt/kvplace/.venv"
PY = f"{VENV}/bin/python"
PORT = 8000

app = modal.App("kvplace-scan")
hf_cache = modal.Volume.from_name("kvplace-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("kvplace-results", create_if_missing=True)
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
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "PYTHONPATH": "/experiment",
            "HF_HOME": "/hf",
        }
    )
    .add_local_dir(Path(__file__).parent / "kvplace", "/experiment/kvplace")
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=49152,
    volumes={"/hf": hf_cache, "/results": results_vol},
    timeout=3600,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def run_scan(run: str, contexts: list[str], max_total: int, repeats: int) -> None:
    saved = Path("/results") / run
    saved.mkdir(parents=True, exist_ok=True)
    max_len = int(max(contexts, key=int)) + 2048
    log_path = saved / "server.log"
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [f"{VENV}/bin/vllm", "serve", MODEL, "--port", str(PORT),
             "--block-size", "16", "--max-model-len", str(max_len),
             "--hf-overrides", f'{{"max_position_embeddings": {max_len}}}',
             "--enable-prompt-tokens-details"],
            stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip
    done = threading.Event()

    def commit_periodically() -> None:
        while not done.wait(60):
            results_vol.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        deadline = time.time() + 1500
        while True:
            if server.poll() is not None:
                raise RuntimeError("vLLM server exited during startup")
            if time.time() > deadline:
                raise RuntimeError("vLLM server did not become healthy")
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(5)
        hf_cache.commit()
        subprocess.run(
            [PY, "-m", "kvplace.decode_scan", "--model", MODEL,
             "--contexts", *contexts, "--max-total", str(max_total),
             "--repeats", str(repeats), "--out", str(saved / "scan.jsonl")],
            check=True,
        )  # fmt: skip
    finally:
        done.set()
        server.terminate()
        server.wait(timeout=120)
        results_vol.commit()


@app.local_entrypoint()
def main(
    run: str,
    max_total: int,
    contexts: str = "2000 8000 32000 80000",
    repeats: int = 2,
):
    run_scan.remote(run, contexts.split(), max_total, repeats)
