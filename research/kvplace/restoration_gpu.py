"""First-stage H2D restoration crossover probe; requires CUDA and vLLM."""

import inspect
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path


class CopyTraffic:
    """Bounded offered H2D bursts on a separate CUDA stream."""

    def __init__(self, torch, enabled):
        self.torch = torch
        self.enabled = enabled
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.bytes = 0
        self.started = 0.0
        self.finished = 0.0
        self.host = torch.zeros(128 * 1024**2, dtype=torch.uint8, pin_memory=True)
        self.device = torch.empty_like(self.host, device="cuda")
        self.stream = torch.cuda.Stream()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self.started = time.perf_counter()
            with self.torch.cuda.stream(self.stream):
                # Same offered burst count in each enabled arm. Stop is only
                # for failure cleanup; the caller normally waits for all bursts.
                for _ in range(32):
                    if self.stop.is_set():
                        break
                    for _ in range(8):
                        self.device.copy_(self.host, non_blocking=True)
                        self.bytes += self.host.numel()
                    self.ready.set()
                    self.stream.synchronize()
            self.finished = time.perf_counter()
        except BaseException as exc:
            self.error = repr(exc)
            self.ready.set()

    def start(self):
        if self.enabled:
            self.thread.start()
            if not self.ready.wait(10):
                raise RuntimeError("contention process did not start")

    def finish(self):
        if self.enabled:
            self.thread.join(15)
            if self.thread.is_alive():
                self.stop.set()
                raise RuntimeError("CUDA copy thread hung; abort this GPU job")
            if self.error:
                raise RuntimeError(self.error)


def source_counts(llm):
    llm.llm_engine.do_log_stats()
    counts = {}
    for metric in llm.get_metrics():
        if metric.name == "vllm:prompt_tokens_by_source":
            source = metric.labels["source"]
            counts[source] = counts.get(source, 0) + metric.value
    if not counts:
        raise RuntimeError("required prompt-source counters are unavailable")
    return counts


def main():
    import torch
    from huggingface_hub import model_info

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no simulated results are produced")
    if "max_load_tokens" not in inspect.getsource(scheduler):
        raise RuntimeError("installed vLLM lacks native per-request load caps")
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    revision = model_info(model).sha
    metadata = {
        "stage": "H2D_only_crossover_probe",
        "model": model,
        "model_revision": revision,
        "vllm": vllm.__version__,
        "load_cap_backport": "75dc5882699be65eaee144cb7d0b06c1f86d3cec",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "topology": subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
        ).stdout,
        "dtype": "bfloat16",
        "prefix_lengths": [4096, 8192],
        "append_tokens": 64,
        "repetitions": 3,
        "seed": 71,
        "note": "Synthetic token IDs; no real agent-content replay or p95 claim",
    }
    llm = LLM(
        model=model,
        revision=revision,
        dtype="bfloat16",
        max_model_len=8272,
        max_num_seqs=1,
        max_num_batched_tokens=8256,
        gpu_memory_utilization=0.65,
        num_gpu_blocks_override=1024,
        block_size=16,
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 1024**3,
                "block_size": 16,
                "store_threshold": 0,
            },
        ),
    )
    rows = []
    params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    rng = random.Random(71)
    prompt = [rng.randrange(1000, 20000) for _ in range(8256)]
    for length in (4096, 8192):
        llm.generate(
            {"prompt_token_ids": prompt[: length + 64]}, params, use_tqdm=False
        )
        for cap in (None, 0, length // 4, length // 2, length * 3 // 4):
            if not llm.reset_prefix_cache(reset_connector=True):
                raise RuntimeError("warmup full cache reset failed")
            llm.generate({"prompt_token_ids": prompt[:length]}, params, use_tqdm=False)
            if not llm.reset_prefix_cache(reset_connector=False):
                raise RuntimeError("warmup GPU cache reset failed")
            extra = (
                {} if cap is None else {"kv_transfer_params": {"max_load_tokens": cap}}
            )
            llm.generate(
                {"prompt_token_ids": prompt[: length + 64]},
                SamplingParams(
                    temperature=0, max_tokens=1, ignore_eos=True, extra_args=extra
                ),
                use_tqdm=False,
            )
    deadline = time.monotonic() + 240
    references = {}
    for length in (4096, 8192):
        for enabled in (False, True):
            traffic = CopyTraffic(torch, enabled)
            for repetition in range(3):
                arms = [
                    ("reload", None),
                    ("recompute", 0),
                    ("quarter", length // 4),
                    ("half", length // 2),
                    ("three_quarters", length * 3 // 4),
                ]
                rng.shuffle(arms)
                for arm, cap in arms:
                    if time.monotonic() > deadline:
                        raise RuntimeError("probe exceeded internal time budget")
                    if not llm.reset_prefix_cache(reset_connector=True):
                        raise RuntimeError("full cache reset failed")
                    llm.generate(
                        {"prompt_token_ids": prompt[:length]}, params, use_tqdm=False
                    )
                    if not llm.reset_prefix_cache(reset_connector=False):
                        raise RuntimeError("GPU-only cache reset failed")
                    before = source_counts(llm)
                    extra = (
                        {}
                        if cap is None
                        else {"kv_transfer_params": {"max_load_tokens": cap}}
                    )
                    sample_params = SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        ignore_eos=True,
                        extra_args=extra,
                    )
                    traffic.bytes = 0
                    traffic.thread = threading.Thread(target=traffic._run, daemon=True)
                    traffic.ready.clear()
                    try:
                        traffic.start()
                        start = time.perf_counter()
                        output = llm.generate(
                            {"prompt_token_ids": prompt[: length + 64]},
                            sample_params,
                            use_tqdm=False,
                        )[0]
                        elapsed = time.perf_counter() - start
                    finally:
                        traffic.finish()
                    after = source_counts(llm)
                    delta = {k: after[k] - before.get(k, 0) for k in after}
                    expected = length if cap is None else cap
                    actual = delta.get("external_kv_transfer", 0)
                    if actual != expected or delta.get("local_cache_hit", 0):
                        raise RuntimeError(
                            f"unmatched cache state: expected {expected}, {delta}"
                        )
                    if delta.get("local_compute") != length + 64 - expected:
                        raise RuntimeError(f"unexpected recompute accounting: {delta}")
                    tokens = list(output.outputs[0].token_ids)
                    references.setdefault(length, tokens)
                    if tokens != references[length]:
                        raise RuntimeError("greedy output differs across arms")
                    row = {
                        "prefix_tokens": length,
                        "contention": enabled,
                        "repetition": repetition,
                        "arm": arm,
                        "one_token_completion_s": elapsed,
                        "engine_ttft_s": output.metrics.first_token_latency
                        if output.metrics
                        else None,
                        "prompt_sources": delta,
                        "output_token_ids": tokens,
                        "background_bytes": traffic.bytes,
                        "background_s": traffic.finished - traffic.started
                        if enabled
                        else 0,
                        "background_overlap_s": max(
                            0.0,
                            min(start + elapsed, traffic.finished)
                            - max(start, traffic.started),
                        )
                        if enabled
                        else 0.0,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    Path("/tmp/restoration_gpu.json").write_text(
                        json.dumps({"metadata": metadata, "rows": rows}, indent=2)
                    )
    summary = []
    for length in (4096, 8192):
        for enabled in (False, True):
            for arm in ("reload", "recompute", "quarter", "half", "three_quarters"):
                values = [
                    r["one_token_completion_s"]
                    for r in rows
                    if r["prefix_tokens"] == length
                    and r["contention"] == enabled
                    and r["arm"] == arm
                ]
                summary.append(
                    {
                        "prefix_tokens": length,
                        "contention": enabled,
                        "arm": arm,
                        "median_s": statistics.median(values),
                    }
                )
    Path("/tmp/restoration_gpu.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows, "summary": summary}, indent=2)
    )


if __name__ == "__main__":
    main()
