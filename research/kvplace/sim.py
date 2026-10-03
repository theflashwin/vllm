"""Offline block-level simulator for KV placement policies (week 2 go/no-go).

Models one vLLM instance with three KV tiers — GPU prefix cache, CPU primary
offload tier, secondary tier (e.g. local NVMe FS) — and replays a trace in
simulated time. It mirrors the vLLM data path that matters for placement:

  * Lookup returns the longest contiguous prefix of full blocks found in any
    tier; the first miss ends reuse (prefix hashing).
  * CPU is the only tier with GPU access: secondary hits are staged
    secondary->CPU->GPU and promoted into CPU.
  * Newly computed prompt blocks are stored to CPU immediately and cascaded to
    the secondary tier (write-through), unless the policy skips the write.
  * Prefill is serialized on one GPU (FIFO); decode does not contend.

Deliberate simplifications (see README): no GPU working-set pinning, cache
state updates are applied at arrival, transfer bandwidth is uncontended.

Example:
    python -m kvplace.sim trace.jsonl --gpu-blocks 4000 --cpu-blocks 16000 \
        --policies lru reuse --predictor oracle

"""

import argparse
import heapq
import json
import math
import statistics
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from kvplace.hints import ReuseHint
from kvplace.predictors import Predictor, get_predictor, is_post_response
from kvplace.trace import Session, read_trace

BlockKey = tuple[str, int]  # (session_id or "shared", block index)
SHARED = "shared"


@dataclass
class CostModel:
    block_tokens: int = 16
    kv_bytes_per_token: int = 57344  # Qwen2.5-7B bf16: 2*28*4*128*2
    gpu_cpu_gbps: float = 20.0  # effective, per direction
    gpu_cpu_lat_s: float = 20e-6  # per transfer job
    sec_cpu_gbps: float = 3.0
    sec_cpu_lat_s: float = 200e-6
    prefill_tok_per_s: float = 12000.0
    decode_s_per_tok: float = 0.02
    sched_overhead_s: float = 0.005

    @property
    def block_bytes(self) -> int:
        return self.block_tokens * self.kv_bytes_per_token

    def cpu_to_gpu_s(self, n: int) -> float:
        if n == 0:
            return 0.0
        return self.gpu_cpu_lat_s + n * self.block_bytes / (self.gpu_cpu_gbps * 1e9)

    def sec_to_cpu_s(self, n: int) -> float:
        if n == 0:
            return 0.0
        return self.sec_cpu_lat_s + n * self.block_bytes / (self.sec_cpu_gbps * 1e9)

    @classmethod
    def from_json(cls, path: str) -> "CostModel":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LRUTier:
    """Block-level LRU. Touch blocks tail-first so the prefix head is the most
    recent (matches vLLM's LRUCachePolicy and GPU free-queue ordering)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.blocks: OrderedDict[BlockKey, None] = OrderedDict()

    def __contains__(self, k: BlockKey) -> bool:
        return k in self.blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def touch(self, keys: Iterable[BlockKey], now: float) -> None:
        for k in reversed(list(keys)):
            if k in self.blocks:
                self.blocks.move_to_end(k)

    def insert(self, keys: list[BlockKey], now: float) -> list[BlockKey]:
        """Insert (or refresh) keys; return evicted keys."""
        if self.capacity <= 0:
            return []
        evicted = []
        for k in reversed(keys):
            if k in self.blocks:
                self.blocks.move_to_end(k)
                continue
            if len(self.blocks) >= self.capacity:
                evicted.append(self.blocks.popitem(last=False)[0])
            self.blocks[k] = None
        return evicted

    def set_hint(self, session_id: str, hint: ReuseHint | None, now: float) -> None:
        pass


class ReuseAwareTier(LRUTier):
    """Evicts the session whose predicted next use is farthest in the future,
    tail blocks first. Approximates Belady at session granularity.

    Scores (predicted next-use time):
      * session marked final           -> +inf (evict first)
      * hinted reuse at t, not overdue -> t
      * overdue by d                   -> now + d (long tools tend to run long)
      * no hint                        -> last_use + default_horizon_s
      * shared blocks                  -> -inf (keep; evicted only via LRU
                                          fallback when nothing else is left)
    """

    def __init__(self, capacity: int, default_horizon_s: float = 30.0):
        super().__init__(capacity)
        self.default_horizon_s = default_horizon_s
        self.deadline: dict[str, float] = {}
        self.final: set[str] = set()
        self.last_use: dict[str, float] = {}
        self.by_session: dict[str, set[int]] = defaultdict(set)

    def _score(self, sid: str, now: float) -> float:
        if sid == SHARED:
            return -math.inf
        if sid in self.final:
            return math.inf
        if sid in self.deadline:
            t = self.deadline[sid]
            return t if t >= now else now + (now - t)
        return self.last_use.get(sid, now) + self.default_horizon_s

    def set_hint(self, session_id: str, hint: ReuseHint | None, now: float) -> None:
        self.last_use[session_id] = now
        self.deadline.pop(session_id, None)
        if hint is None:
            return
        if hint.final:
            self.final.add(session_id)
        elif hint.expected_reuse_s is not None:
            self.deadline[session_id] = now + hint.expected_reuse_s

    def touch(self, keys: Iterable[BlockKey], now: float) -> None:
        super().touch(keys, now)
        for sid, _ in keys:
            self.last_use[sid] = now

    def insert(self, keys: list[BlockKey], now: float) -> list[BlockKey]:
        if self.capacity <= 0:
            return []
        new = [k for k in dict.fromkeys(keys) if k not in self.blocks]
        self.touch([k for k in keys if k in self.blocks], now)
        need = len(self.blocks) + len(new) - self.capacity
        protect = {k[0] for k in keys}
        evicted = self._evict(need, now, protect) if need > 0 else []
        for k in reversed(new[: self.capacity]):
            self.blocks[k] = None
            self.by_session[k[0]].add(k[1])
        return evicted

    def _evict(self, n: int, now: float, protect: set[str]) -> list[BlockKey]:
        order = sorted(
            (s for s in self.by_session if s not in protect and self.by_session[s]),
            key=lambda s: self._score(s, now),
            reverse=True,
        )
        evicted: list[BlockKey] = []
        for sid in order:
            idxs = sorted(self.by_session[sid], reverse=True)
            for i in idxs[: n - len(evicted)]:
                self._drop((sid, i))
                evicted.append((sid, i))
            if len(evicted) >= n:
                return evicted
        # Fall back to LRU over whatever is left (including protected).
        while len(evicted) < n and self.blocks:
            k = next(iter(self.blocks))
            self._drop(k)
            evicted.append(k)
        return evicted

    def _drop(self, k: BlockKey) -> None:
        del self.blocks[k]
        self.by_session[k[0]].discard(k[1])


@dataclass
class PolicyConfig:
    name: str
    cpu_policy: str = "lru"  # lru | reuse
    use_hints: bool = False
    prefetch: bool = False
    # Selective write: never cascade final-turn blocks, and defer cascading
    # blocks whose predicted reuse is sooner than write_defer_horizon_s; a
    # deferred block is written back to secondary only if CPU evicts it first.
    selective_write: bool = False
    write_defer_horizon_s: float = 5.0
    prefetch_margin_s: float = 0.5


POLICIES: dict[str, PolicyConfig] = {
    "lru": PolicyConfig("lru"),
    "reuse_evict": PolicyConfig("reuse_evict", "reuse", use_hints=True),
    "reuse_prefetch": PolicyConfig(
        "reuse_prefetch", "lru", use_hints=True, prefetch=True
    ),
    "reuse": PolicyConfig(
        "reuse", "reuse", use_hints=True, prefetch=True, selective_write=True
    ),
}


@dataclass
class TurnRecord:
    session_id: str
    turn: int
    arrival_s: float
    ttft_s: float
    prompt_tokens: int
    gpu_hit_blocks: int
    cpu_hit_blocks: int
    sec_hit_blocks: int
    miss_blocks: int
    # Full blocks this turn could have reused (the previous turn's prompt, or
    # the shared prefix on turn 0). Misses beyond this are unavoidable.
    reusable_blocks: int
    gap_s: float  # tool duration that preceded this turn (0 for turn 0)


@dataclass
class Counters:
    cpu_to_gpu_blocks: int = 0
    sec_to_cpu_demand_blocks: int = 0
    sec_to_cpu_prefetch_blocks: int = 0
    gpu_to_cpu_store_blocks: int = 0
    cpu_to_sec_write_blocks: int = 0
    cpu_to_sec_write_skipped: int = 0
    cpu_to_sec_writeback_blocks: int = 0
    prefetch_used_blocks: int = 0
    prefetch_wasted_blocks: int = 0
    hint_requests: int = 0


@dataclass
class SimResult:
    policy: str
    predictor: str
    records: list[TurnRecord] = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)

    def summary(self) -> dict:
        resume = [r.ttft_s for r in self.records if r.turn > 0]
        first = [r.ttft_s for r in self.records if r.turn == 0]
        blocks = defaultdict(int)
        for r in self.records:
            if r.turn == 0:
                continue
            blocks["gpu"] += r.gpu_hit_blocks
            blocks["cpu"] += r.cpu_hit_blocks
            blocks["sec"] += r.sec_hit_blocks
            blocks["lost"] += max(
                0,
                r.reusable_blocks
                - r.gpu_hit_blocks
                - r.cpu_hit_blocks
                - r.sec_hit_blocks,
            )
        total = sum(blocks.values()) or 1
        return {
            "policy": self.policy,
            "predictor": self.predictor,
            "turns": len(self.records),
            "resume_ttft_mean": statistics.fmean(resume) if resume else 0.0,
            "resume_ttft_p50": _pct(resume, 50),
            "resume_ttft_p90": _pct(resume, 90),
            "resume_ttft_p99": _pct(resume, 99),
            "first_ttft_p50": _pct(first, 50),
            **{f"resume_frac_{k}": blocks[k] / total for k in blocks},
            **asdict(self.counters),
        }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def _block_keys(session: Session, n_tokens: int, block_tokens: int) -> list[BlockKey]:
    keys = []
    for i in range(n_tokens // block_tokens):
        shared = (i + 1) * block_tokens <= session.shared_prefix_tokens
        keys.append((SHARED, i) if shared else (session.session_id, i))
    return keys


class Simulator:
    def __init__(
        self,
        sessions: list[Session],
        policy: PolicyConfig,
        predictor_name: str,
        cost: CostModel,
        gpu_blocks: int,
        cpu_blocks: int,
        sec_blocks: int,
        time_scale: float = 1.0,
    ):
        self.sessions = {s.session_id: s for s in sessions}
        self.policy = policy
        self.predictor: Predictor = get_predictor(
            predictor_name if policy.use_hints else "none"
        )
        self.cost = cost
        self.time_scale = time_scale
        self.gpu = LRUTier(gpu_blocks)
        reuse_cpu = policy.cpu_policy == "reuse"
        self.cpu = ReuseAwareTier(cpu_blocks) if reuse_cpu else LRUTier(cpu_blocks)
        self.sec = LRUTier(sec_blocks)
        self.gpu_free_at = 0.0
        self.gaps: dict[str, list[float]] = defaultdict(list)
        self.prefetched: set[BlockKey] = set()
        self.deferred: set[BlockKey] = set()
        self.result = SimResult(
            policy.name, predictor_name if policy.use_hints else "none"
        )
        self._events: list[tuple[float, int, str, str, int]] = []
        self._seq = 0

    def _push(self, t: float, kind: str, sid: str, k: int) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, sid, k))

    def run(self) -> SimResult:
        for s in self.sessions.values():
            self._push(s.start_s * self.time_scale, "arrive", s.session_id, 0)
        while self._events:
            t, _, kind, sid, k = heapq.heappop(self._events)
            if kind == "arrive":
                self._arrive(t, self.sessions[sid], k)
            elif kind == "prefetch":
                self._prefetch(t, self.sessions[sid], k)
        # Prefetched but never read before the trace ended.
        self.result.counters.prefetch_wasted_blocks += len(self.prefetched)
        return self.result

    def _note_cpu_evictions(self, evicted: list[BlockKey], now: float) -> None:
        writeback = []
        for k in evicted:
            if k in self.prefetched:
                self.prefetched.discard(k)
                self.result.counters.prefetch_wasted_blocks += 1
            if k in self.deferred:
                self.deferred.discard(k)
                writeback.append(k)
        if writeback:
            self.sec.insert(writeback, now)
            self.result.counters.cpu_to_sec_writeback_blocks += len(writeback)

    def _arrive(self, now: float, s: Session, k: int) -> None:
        c = self.result.counters
        cost = self.cost
        prompt_len = s.prompt_lengths()[k]
        keys = _block_keys(s, prompt_len, cost.block_tokens)
        if k > 0 and hasattr(self.predictor, "observe"):
            # The previous tool call just completed.
            prev = s.turns[k - 1]
            self.predictor.observe(prev.tool_name, prev.tool_duration_s)
        hint = self._scaled(self.predictor(s, k, self.gaps[s.session_id]))
        # Post-response hints arrive via a hint-only request after the
        # response, so store-time decisions (selective write) can't use them.
        post = is_post_response(self.predictor)
        submit_hint = None if post else hint
        if post and hint is not None:
            c.hint_requests += 1

        # Lookup: longest contiguous prefix across tiers.
        tiers = []
        for key in keys:
            if key in self.gpu:
                tiers.append("gpu")
            elif key in self.cpu:
                tiers.append("cpu")
            elif key in self.sec:
                tiers.append("sec")
            else:
                break
        n_gpu = tiers.count("gpu")
        n_cpu = tiers.count("cpu")
        n_sec = tiers.count("sec")
        n_hit = len(tiers)
        hit_keys = keys[:n_hit]
        sec_keys = [key for key, t in zip(hit_keys, tiers) if t == "sec"]
        cpu_keys = [key for key, t in zip(hit_keys, tiers) if t == "cpu"]

        for key in cpu_keys:
            if key in self.prefetched:
                self.prefetched.discard(key)
                c.prefetch_used_blocks += 1

        load_s = cost.sec_to_cpu_s(n_sec) + cost.cpu_to_gpu_s(n_cpu + n_sec)
        c.sec_to_cpu_demand_blocks += n_sec
        c.cpu_to_gpu_blocks += n_cpu + n_sec

        uncached_tokens = prompt_len - n_hit * cost.block_tokens
        prefill_s = uncached_tokens / cost.prefill_tok_per_s
        start = max(now, self.gpu_free_at)
        self.gpu_free_at = start + prefill_s
        ttft = cost.sched_overhead_s + (start - now) + load_s + prefill_s
        finish = now + ttft + s.turns[k].output_tokens * cost.decode_s_per_tok

        # Cache updates.
        self.gpu.insert(keys, now)
        self.sec.touch(sec_keys, now)
        self._note_cpu_evictions(self.cpu.insert(sec_keys, now), now)
        self.cpu.touch(cpu_keys, now)
        new_keys = keys[n_hit:]
        # Deadlines count from the response finishing (when vLLM's policy
        # sees on_request_finished), not from arrival.
        self.cpu.set_hint(s.session_id, hint, finish)
        self._note_cpu_evictions(self.cpu.insert(new_keys, now), now)
        c.gpu_to_cpu_store_blocks += len(new_keys) if self.cpu.capacity > 0 else 0
        if self.sec.capacity > 0 and new_keys:
            mode = self._secondary_write_mode(submit_hint)
            if mode == "skip":
                c.cpu_to_sec_write_skipped += len(new_keys)
            elif mode == "defer":
                self.deferred.update(k for k in new_keys if k in self.cpu)
            else:
                self.sec.insert(new_keys, now)
                c.cpu_to_sec_write_blocks += len(new_keys)
        if hint is not None and hint.final:
            # Earlier deferred blocks of a finished session need no write-back.
            self.deferred.difference_update(keys)

        self.result.records.append(
            TurnRecord(
                s.session_id,
                k,
                now,
                ttft,
                prompt_len,
                n_gpu,
                n_cpu,
                n_sec,
                len(keys) - n_hit,
                (s.prompt_lengths()[k - 1] if k else s.shared_prefix_tokens)
                // cost.block_tokens,
                s.turns[k - 1].tool_duration_s * self.time_scale if k else 0.0,
            )
        )

        if k + 1 < len(s.turns):
            gap = s.turns[k].tool_duration_s * self.time_scale
            self.gaps[s.session_id].append(s.turns[k].tool_duration_s)
            self._push(finish + gap, "arrive", s.session_id, k + 1)
            if self.policy.prefetch and hint and hint.expected_reuse_s is not None:
                # Promote early enough to hide secondary->CPU latency.
                lead = cost.sec_to_cpu_s(len(keys)) + self.policy.prefetch_margin_s
                t = finish + max(0.0, hint.expected_reuse_s - lead)
                self._push(t, "prefetch", s.session_id, k)

    def _scaled(self, hint: ReuseHint | None) -> ReuseHint | None:
        if hint is None or hint.expected_reuse_s is None or self.time_scale == 1:
            return hint
        return ReuseHint(
            hint.session_id,
            hint.turn,
            hint.expected_reuse_s * self.time_scale,
            hint.final,
        )

    def _secondary_write_mode(self, hint: ReuseHint | None) -> str:
        """Return write (cascade now), defer (write back on CPU eviction) or
        skip (never)."""
        if not self.policy.selective_write or hint is None:
            return "write"
        if hint.final:
            return "skip"
        if (
            hint.expected_reuse_s is not None
            and hint.expected_reuse_s < self.policy.write_defer_horizon_s
        ):
            return "defer"
        return "write"

    def _prefetch(self, now: float, s: Session, k: int) -> None:
        keys = _block_keys(s, s.prompt_lengths()[k], self.cost.block_tokens)
        todo = []
        for key in keys:
            if key in self.gpu or key in self.cpu:
                continue
            if key not in self.sec:
                break
            todo.append(key)
        if not todo:
            return
        self._note_cpu_evictions(self.cpu.insert(todo, now), now)
        self.prefetched.update(key for key in todo if key in self.cpu)
        self.result.counters.sec_to_cpu_prefetch_blocks += len(todo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--policies", nargs="+", default=list(POLICIES))
    p.add_argument(
        "--predictors",
        nargs="+",
        default=["oracle"],
        help="none | oracle | noisy:<sigma> | ewma[:<alpha>]",
    )
    p.add_argument("--gpu-blocks", type=int, default=4000)
    p.add_argument("--cpu-blocks", type=int, default=16000)
    p.add_argument("--sec-blocks", type=int, default=10_000_000)
    p.add_argument("--cost-json", help="CostModel overrides (microbench output)")
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--out", help="write per-turn records + summaries as JSON")
    args = p.parse_args()

    sessions = read_trace(args.trace)
    cost = CostModel.from_json(args.cost_json) if args.cost_json else CostModel()
    runs = []
    for pol in args.policies:
        preds = args.predictors if POLICIES[pol].use_hints else ["none"]
        for pred in preds:
            sim = Simulator(
                sessions,
                POLICIES[pol],
                pred,
                cost,
                args.gpu_blocks,
                args.cpu_blocks,
                args.sec_blocks,
                args.time_scale,
            )
            runs.append(sim.run())

    cols = [
        "policy",
        "predictor",
        "resume_ttft_p50",
        "resume_ttft_p90",
        "resume_ttft_p99",
        "resume_frac_gpu",
        "resume_frac_cpu",
        "resume_frac_sec",
        "resume_frac_lost",
        "sec_to_cpu_prefetch_blocks",
        "prefetch_wasted_blocks",
        "cpu_to_sec_write_blocks",
        "cpu_to_sec_writeback_blocks",
    ]
    summaries = [r.summary() for r in runs]
    print("\t".join(cols))
    for sm in summaries:
        vals = [sm.get(c, 0) for c in cols]
        print("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "cost": asdict(cost),
                    "summaries": summaries,
                    "records": {
                        f"{r.policy}/{r.predictor}": [asdict(x) for x in r.records]
                        for r in runs
                    },
                },
                f,
            )


if __name__ == "__main__":
    main()
