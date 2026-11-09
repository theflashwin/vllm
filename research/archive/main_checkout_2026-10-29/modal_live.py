"""Live vLLM replay on Modal L4s: LRU vs ReuseAwareCachePolicy on the CPU tier.

    uvx --from modal modal run research/modal_live.py

Each policy runs in its own L4 container in parallel, with identical server
settings: GPU blocks pinned to the simulator's value, a small CPU tier, and a
write-through filesystem tier on the container disk. The replay client sends
the same requests and hint traffic to both; only the CPU eviction policy
differs. Results land in research/results/live/<policy>/.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PY = "/opt/kvplace/.venv/bin/python"
TRACE = "live70.jsonl"
GPU_BLOCKS = 3800
CPU_BLOCKS = 4000
BLOCK_BYTES = 16 * 57344
TIME_SCALE = 0.2
PREDICTOR = "tool-nfh"
POLICIES = {"lru": "lru", "ours": "ReuseAwareCachePolicy"}

HERE = Path(__file__).parent
app = modal.App("kvplace-live")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.11.8")
    .run_commands(
        "uv venv /opt/kvplace/.venv --python 3.12",
        f"uv pip install --python {PY} vllm==0.31.0 aiohttp",
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTHONPATH": "/experiment"})
    .add_local_dir(HERE / "kvplace", "/experiment/kvplace")
    .add_local_file(HERE / "traces" / TRACE, f"/experiment/{TRACE}")
)


def _wait_healthy(server: subprocess.Popen) -> None:
    for _ in range(240):
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM server did not become healthy")


def _replay(trace: str, out: str, time_scale: float, max_sessions: int | None):
    cmd = [PY, "-m", "kvplace.replay", trace, "--predictor", PREDICTOR,
           "--time-scale", str(time_scale), "--out", out]  # fmt: skip
    if max_sessions:
        cmd += ["--max-sessions", str(max_sessions)]
    subprocess.run(cmd, check=True, timeout=3000)


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 8),
    memory=(32768, 49152),
    timeout=3600,
    startup_timeout=600,
    retries=0,
    max_containers=2,
    scaledown_window=2,
)
def run(name: str) -> dict:
    free_gb = shutil.disk_usage("/tmp").free / 1e9
    if free_gb < 70:
        return {"policy": name, "error": f"only {free_gb:.0f} GB free on /tmp"}
    extra = {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": CPU_BLOCKS * BLOCK_BYTES,
        "eviction_policy": POLICIES[name],
        "cache_policy_module_path": "kvplace.vllm_policy",
        "secondary_tiers": [{"type": "fs", "root_dir": "/tmp/kvfs"}],
    }
    kv_config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }
    log = open("/tmp/server.log", "w")  # noqa: SIM115
    server = subprocess.Popen(
        [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
         "--block-size", "16", "--max-model-len", "32768",
         "--num-gpu-blocks-override", str(GPU_BLOCKS),
         "--kv-transfer-config", json.dumps(kv_config), "--port", "8000"],
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "KVPLACE_LOG_HINTS": "1"},
    )  # fmt: skip
    trace = f"/experiment/{TRACE}"
    result: dict = {"policy": name, "extra_config": extra}
    try:
        _wait_healthy(server)
        # Smoke test: a few sessions, fast, so config errors fail early.
        _replay(trace, "/tmp/smoke", 0.01, 3)
        hints = Path("/tmp/server.log").read_text().count("kvplace hint")
        result["smoke_hints_logged"] = hints
        if name == "ours" and hints == 0:
            raise RuntimeError("no hints reached the policy")
        _replay(trace, "/tmp/out", TIME_SCALE, None)
        out = Path("/tmp/out")
        result["requests_csv"] = (out / "requests.csv").read_text()
        result["metrics"] = json.loads((out / "metrics_delta.json").read_text())
        result["hints_logged"] = (
            Path("/tmp/server.log").read_text().count("kvplace hint") - hints
        )
    except Exception as e:
        result["error"] = repr(e)
    finally:
        server.terminate()
        server.wait(timeout=60)
        log.close()
        lines = Path("/tmp/server.log").read_text().splitlines()
        result["server_log_tail"] = "\n".join(
            ln for ln in lines[-400:] if "kvplace hint" not in ln
        )[-20000:]
    return result


@app.local_entrypoint()
def main(policies: str = "lru,ours"):
    names = policies.split(",")
    for res in run.map(names):
        out = HERE / "results" / "live" / res["policy"]
        out.mkdir(parents=True, exist_ok=True)
        if "requests_csv" in res:
            (out / "requests.csv").write_text(res.pop("requests_csv"))
        (out / "server_log_tail.txt").write_text(res.pop("server_log_tail", ""))
        (out / "result.json").write_text(json.dumps(res, indent=1))
        print(res["policy"], res.get("error", "ok"), res.get("hints_logged"))
