"""One-shot, bounded GPU probe: modal run research/modal_restoration.py."""

import json
import subprocess
from pathlib import Path

import modal

app = modal.App("kvplace-restoration-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("uv==0.11.8")
    .run_commands(
        "uv venv /opt/kvplace/.venv --python 3.12",
        "uv pip install --python /opt/kvplace/.venv/bin/python vllm==0.30.0",
    )
    .apt_install("git")
    .add_local_file(
        Path(__file__).parent / "scripts" / "native_load_cap.patch",
        "/experiment/load_cap.patch",
        copy=True,
    )
    .run_commands(
        "cd /opt/kvplace/.venv/lib/python3.12/site-packages && "
        "git apply --check /experiment/load_cap.patch && "
        "git apply /experiment/load_cap.patch",
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_file(
        Path(__file__).parent / "kvplace" / "restoration_gpu.py",
        "/experiment/restoration_gpu.py",
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=(2, 2),
    memory=(16384, 24576),
    timeout=480,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def probe():
    result = subprocess.run(
        ["/opt/kvplace/.venv/bin/python", "/experiment/restoration_gpu.py"],
        timeout=450,
    )
    path = Path("/tmp/restoration_gpu.json")
    return {
        "exit_code": result.returncode,
        "measurements": json.loads(path.read_text()) if path.exists() else None,
    }


@app.local_entrypoint()
def main(out: str = "/tmp/kvplace-restoration-modal.json"):
    result = probe.remote()
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"Saved GPU probe: {out}; exit code: {result['exit_code']}")
    if result["exit_code"]:
        raise RuntimeError("GPU probe failed; inspect logs and saved partial results")
