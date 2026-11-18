"""One row per arm of finished live runs: TTFT, disk reads, recompute, hints.

TTFT covers resumed turns (turn > 0). The p90/p99 intervals are 95% bootstrap
intervals over sessions. Prints tab-separated rows.

    uv run -q --no-project --python 3.12 --with numpy \
        python analysis/live_table.py results/live/live70_ts0.35_*
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)


def load(arm: Path) -> list[dict]:
    with open(arm / "requests.csv") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["kind"] == "turn" and int(r["turn"]) > 0 and r["ttft_s"]
        ]


def metric(m: dict, name: str, **labels) -> float:
    return sum(
        v
        for k, v in m.items()
        if k.split("{")[0] == name
        and all(f'{a}="{b}"' in k for a, b in labels.items())
    )


def boot(rows: list[dict], q: float, n: int = 1000) -> np.ndarray:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(float(r["ttft_s"]))
    sids = list(by)
    out = []
    for _ in range(n):
        pick = rng.choice(len(sids), len(sids))
        out.append(np.quantile(np.concatenate([by[sids[i]] for i in pick]), q))
    return np.quantile(out, [0.025, 0.975])


def main(dirs: list[str]) -> None:
    print(
        "run\tconfig\tpred\tcpuGB\tdisk\tshared\tn\tp50\tp90\tp90CI\tp99\tp99CI"
        "\tmean\tqueue_mean\tdiskreadGB\trecompute_k\tfinal_udp"
    )
    for d in map(Path, dirs):
        a = json.loads((d / "args.json").read_text())
        for cfg in a["configs"]:
            arm = d / cfg
            rows = load(arm)
            x = np.array([float(r["ttft_s"]) for r in rows])
            m = json.loads((arm / "metrics_delta.json").read_text())["metrics"]
            queue = metric(m, "vllm:request_queue_time_seconds_sum") / max(
                1, metric(m, "vllm:request_queue_time_seconds_count")
            )
            read_gb = metric(m, "vllm:kv_offload_tiering_read_bytes_total") / 1e9
            recompute_k = (
                metric(m, "vllm:prompt_tokens_by_source_total", source="local_compute")
                / 1e3
            )
            log = arm / "server.log"
            text = log.read_text(errors="ignore") if log.exists() else ""
            finals = len(re.findall(r"final=True.*\(udp\)", text))
            c90, c99 = boot(rows, 0.9), boot(rows, 0.99)
            print(
                f"{d.name}\t{cfg}\t{a['predictor']}\t{a['cpu_bytes'] / 1e9:.1f}"
                f"\t{a.get('disk_gb_s')}\t{a.get('disk_shared')}\t{len(x)}"
                f"\t{np.quantile(x, 0.5):.3f}\t{np.quantile(x, 0.9):.2f}"
                f"\t{c90[0]:.1f}-{c90[1]:.1f}\t{np.quantile(x, 0.99):.1f}"
                f"\t{c99[0]:.0f}-{c99[1]:.0f}\t{x.mean():.2f}\t{queue:.2f}"
                f"\t{read_gb:.0f}\t{recompute_k:.0f}\t{finals}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
