"""Compare bounded, uncertainty-aware prefetch on a shared secondary-read link.

This is a simulator experiment, not a serving policy. Demand reads and
nonpreemptible prefetch reads serialize on one link; prefetch becomes visible
only on completion. GPU/CPU transfers, writes, decode contention, and cache
updates for demand requests retain the simplifications in kvplace.sim.
"""

import argparse
import heapq
import json
from dataclasses import asdict, dataclass, replace

from kvplace.hints import ReuseHint
from kvplace.sim import CostModel, PolicyConfig, Simulator, _block_keys, _pct
from kvplace.trace import Session, read_trace


@dataclass
class PrefetchConfig:
    bandwidth_fraction: float = 0.1
    capacity_fraction: float = 0.1
    chunk_blocks: int = 256
    residency_window_s: float = 2.0
    min_probability: float = 0.5
    min_samples: int = 8
    quantile: float = 0.5

    def __post_init__(self):
        if not 0 < self.bandwidth_fraction <= 1:
            raise ValueError("bandwidth_fraction must be in (0, 1]")
        if not 0 < self.capacity_fraction <= 1:
            raise ValueError("capacity_fraction must be in (0, 1]")
        if self.chunk_blocks < 1 or self.min_samples < 1:
            raise ValueError("chunk_blocks and min_samples must be positive")
        if self.residency_window_s <= 0:
            raise ValueError("residency_window_s must be positive")
        if not 0 <= self.min_probability <= 1 or not 0 <= self.quantile <= 1:
            raise ValueError("probability and quantile must be in [0, 1]")


class PrefetchExperiment(Simulator):
    """Keep CPU eviction and hint traffic identical across prefetch variants."""

    def __init__(self, sessions, variant, cost, gpu_blocks, cpu_blocks, config):
        if variant not in ("demand", "fixed", "budget", "uncertain"):
            raise ValueError(f"unknown variant {variant!r}")
        super().__init__(
            sessions,
            PolicyConfig(variant, prefetch=variant != "demand"),
            "tool",
            cost,
            gpu_blocks,
            cpu_blocks,
            10_000_000,
        )
        self.config = config
        self.link_free_at = 0.0
        self.response_at: dict[tuple[str, int], float] = {}
        self.samples: dict[tuple[str, int], list[float]] = {}
        self.jobs: dict[tuple[str, int], list[tuple[str, int]]] = {}
        self.pending: set[tuple[str, int]] = set()
        self.credit = float(config.chunk_blocks)
        self.credit_at = 0.0
        self.metrics = {
            "prefetch_jobs": 0,
            "prefetch_rejected_busy": 0,
            "prefetch_rejected_budget": 0,
            "prefetch_rejected_uncertain": 0,
            "prefetch_rejected_cold": 0,
            "prefetch_stale_blocks": 0,
            "demand_link_wait_s": 0.0,
            "prefetch_link_busy_s": 0.0,
            "peak_prefetch_blocks": 0,
        }
        self.probability_log: list[dict] = []

    def _push(self, t, kind, sid, k):
        if kind == "hint_request":
            self.response_at[sid, k] = t
        super()._push(t, kind, sid, k)

    def _secondary_load_s(self, now, blocks):
        if not blocks:
            return 0.0
        start = max(now, self.link_free_at)
        self.metrics["demand_link_wait_s"] += start - now
        self.link_free_at = start + self.cost.sec_to_cpu_s(blocks)
        return self.link_free_at - now

    def _schedule_prefetch(self, now: float, s: Session, k: int, hint: ReuseHint):
        history = list(self.predictor.history.get(s.turns[k].tool_name, []))
        self.samples[s.session_id, k] = history
        if self.policy.name == "uncertain" and len(history) < self.config.min_samples:
            self.metrics["prefetch_rejected_cold"] += 1
            return
        gap = (
            _pct(history, self.config.quantile * 100)
            if self.policy.name == "uncertain"
            else _pct(history, 50)
            if history
            else hint.expected_reuse_s
        )
        keys = _block_keys(s, s.prompt_lengths()[k], self.cost.block_tokens)
        count = len(keys)
        if self.policy.name in ("budget", "uncertain"):
            count = min(count, self.config.chunk_blocks)
        lead = self.cost.sec_to_cpu_s(count) + self.policy.prefetch_margin_s
        # Tool execution began at the original response, before hint delivery.
        target = self.response_at[s.session_id, k] + gap - lead
        self._push(max(now, target), "prefetch", s.session_id, k)

    def _prefetch(self, now: float, s: Session, k: int):
        sid = s.session_id
        if self._latest_turn.get(sid) != k or (sid, k) in self.jobs:
            return
        keys = _block_keys(s, s.prompt_lengths()[k], self.cost.block_tokens)
        todo = []
        for key in keys:
            if key in self.gpu or key in self.cpu:
                continue
            if key not in self.sec:
                break
            # Don't fetch a tail beyond an in-flight missing prefix block.
            if key in self.pending:
                break
            todo.append(key)
        if not todo:
            return
        bounded = self.policy.name in ("budget", "uncertain")
        if bounded:
            if self.link_free_at > now:
                self.metrics["prefetch_rejected_busy"] += 1
                return
            rate = (
                self.config.bandwidth_fraction
                * self.cost.sec_cpu_gbps
                * 1e9
                / self.cost.block_bytes
            )
            self.credit = min(
                self.config.chunk_blocks, self.credit + (now - self.credit_at) * rate
            )
            self.credit_at = now
            capacity = int(self.cpu.capacity * self.config.capacity_fraction)
            available = capacity - len(self.prefetched) - len(self.pending)
            count = min(
                len(todo), self.config.chunk_blocks, int(self.credit), available
            )
            if count <= 0:
                self.metrics["prefetch_rejected_budget"] += 1
                return
            todo = todo[:count]
        if self.policy.name == "uncertain":
            elapsed = max(0.0, now - self.response_at[sid, k])
            remaining = [x for x in self.samples[sid, k] if x > elapsed]
            window = self.cost.sec_to_cpu_s(len(todo)) + self.config.residency_window_s
            probability = (
                sum(x <= elapsed + window for x in remaining) / len(remaining)
                if remaining
                else 0.0
            )
            self.probability_log.append(
                {
                    "session_id": sid,
                    "turn": k,
                    "elapsed_s": elapsed,
                    "probability": probability,
                    "samples": len(remaining),
                    "window_s": window,
                    "admitted": probability >= self.config.min_probability,
                }
            )
            if probability < self.config.min_probability:
                self.metrics["prefetch_rejected_uncertain"] += 1
                return
        if bounded:
            self.credit -= len(todo)
        start = max(now, self.link_free_at)
        duration = self.cost.sec_to_cpu_s(len(todo))
        self.link_free_at = start + duration
        self.jobs[sid, k] = todo
        self.pending.update(todo)
        self.metrics["prefetch_jobs"] += 1
        self.metrics["prefetch_link_busy_s"] += duration
        self.metrics["peak_prefetch_blocks"] = max(
            self.metrics["peak_prefetch_blocks"],
            len(self.pending) + len(self.prefetched),
        )
        self.result.counters.sec_to_cpu_prefetch_blocks += len(todo)
        self._push(self.link_free_at, "prefetch_done", sid, k)

    def _complete_prefetch(self, now, s, k):
        todo = self.jobs.pop((s.session_id, k))
        self.pending.difference_update(todo)
        # A resumed request can make a queued prefetch redundant before it lands.
        usable = [
            key
            for key in todo
            if self._latest_turn.get(s.session_id) == k
            and key not in self.cpu
            and key not in self.gpu
        ]
        stale = len(todo) - len(usable)
        self.metrics["prefetch_stale_blocks"] += stale
        self.result.counters.prefetch_wasted_blocks += stale
        self._note_cpu_evictions(self.cpu.insert(usable, now), now)
        resident = [key for key in usable if key in self.cpu]
        self.prefetched.update(resident)
        self.result.counters.prefetch_wasted_blocks += len(usable) - len(resident)
        if usable and self.policy.name in ("budget", "uncertain"):
            self._push(now, "prefetch", s.session_id, k)

    def run(self):
        for s in self.sessions.values():
            self._push(s.start_s, "arrive", s.session_id, 0)
        while self._events:
            now, _, kind, sid, k = heapq.heappop(self._events)
            s = self.sessions[sid]
            if kind == "prefetch_done":
                self._complete_prefetch(now, s, k)
            else:
                {
                    "arrive": self._arrive,
                    "hint_request": self._hint_request,
                    "hint_update": self._hint_update,
                    "restored": self._restored,
                    "prefetch": self._prefetch,
                }[kind](now, s, k)
        self.result.counters.prefetch_wasted_blocks += len(self.prefetched)
        return self.result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument(
        "--variants", nargs="+", default=["demand", "fixed", "budget", "uncertain"]
    )
    p.add_argument("--gpu-blocks", type=int, default=12000)
    p.add_argument("--cpu-blocks", type=int, default=24000)
    p.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="multiply session start times; tool durations stay unchanged",
    )
    p.add_argument("--bandwidth-fraction", type=float, default=0.1)
    p.add_argument("--capacity-fraction", type=float, default=0.1)
    p.add_argument("--chunk-blocks", type=int, default=256)
    p.add_argument("--residency-window-s", type=float, default=2.0)
    p.add_argument("--min-probability", type=float, default=0.5)
    p.add_argument("--cost-json")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.arrival_scale <= 0:
        p.error("--arrival-scale must be positive")
    config = PrefetchConfig(
        **{
            k: getattr(args, k)
            for k in (
                "bandwidth_fraction",
                "capacity_fraction",
                "chunk_blocks",
                "residency_window_s",
                "min_probability",
            )
        }
    )
    sessions = [
        replace(s, start_s=s.start_s * args.arrival_scale)
        for s in read_trace(args.trace)
    ]
    cost = CostModel.from_json(args.cost_json) if args.cost_json else CostModel()
    summaries, records, decisions = [], {}, {}
    for variant in args.variants:
        sim = PrefetchExperiment(
            sessions, variant, cost, args.gpu_blocks, args.cpu_blocks, config
        )
        result = sim.run()
        summary = result.summary() | sim.metrics
        count = result.counters.sec_to_cpu_prefetch_blocks
        summary["prefetch_useful_fraction"] = (
            result.counters.prefetch_used_blocks / count if count else 0.0
        )
        summary["prefetch_bytes"] = count * cost.block_bytes
        summary["demand_bytes"] = (
            result.counters.sec_to_cpu_demand_blocks * cost.block_bytes
        )
        # Labels are computed after replay; the controller never sees them.
        for decision in sim.probability_log:
            gap = (
                sim.sessions[decision["session_id"]]
                .turns[decision["turn"]]
                .tool_duration_s
            )
            decision["returned_in_window"] = (
                decision["elapsed_s"]
                < gap
                <= decision["elapsed_s"] + decision["window_s"]
            )
        if sim.probability_log:
            summary["probability_brier_score"] = sum(
                (d["probability"] - d["returned_in_window"]) ** 2
                for d in sim.probability_log
            ) / len(sim.probability_log)
        summaries.append(summary)
        records[variant] = [asdict(r) for r in result.records]
        decisions[variant] = sim.probability_log
        print(json.dumps(summary), flush=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "trace": args.trace,
                "cost": asdict(cost),
                "config": asdict(config),
                "gpu_blocks": args.gpu_blocks,
                "cpu_blocks": args.cpu_blocks,
                "arrival_scale": args.arrival_scale,
                "summaries": summaries,
                "records": records,
                "decisions": decisions,
            },
            f,
        )


if __name__ == "__main__":
    main()
