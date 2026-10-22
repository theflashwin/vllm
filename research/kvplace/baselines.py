"""Simple CPU-tier eviction baselines: where does the demotion gain come from?

Each is a drop-in replacement for the CPU tier of `gpu_cpu_aware`: the GPU
tier stays `ReuseAwareTier` fed by the same predictor, so only the CPU->disk
demotion rule changes. Most reuse `ReuseAwareTier`'s machinery (evict whole
sessions, tail blocks first; shared blocks last; sessions of the inserting
request protected) and differ only in the eviction score, where a larger
score is evicted first. Two-stage tiers (`onebit`, `ttl`) first evict whole
sessions matching a predicate, oldest first, then fall back to exactly the
block order of the `lru` tier (the CPU tier of `gpu_aware`):

    slru         idle time (session-granular LRU; a control for the machinery)
    onebit       sessions waiting on a human (latest hint's tool == "human")
                 first, then block LRU; no numeric prediction
    onebit_slru  waiting on a human first, session LRU within each class
    ttl:<T>      sessions idle > T s first, then block LRU
    hazard       expected remaining idle time given elapsed idle time, from
                 all gaps observed so far (no tool information)
    tool_hazard  the same, conditioned on the latest hint's tool
    noaging      `ReuseAwareTier` without aging of overdue hinted sessions
    noaging_all  ... and without aging of unhinted ones (pre-fix behaviour)
    oracle      `ReuseAwareTier` scored with the true next-request time

"Idle" counts from the session's last use as the tier sees it: its last
hint, lookup hit, or store (`ReuseAwareTier.last_use`).
"""

import bisect
import math
from collections import defaultdict
from collections.abc import Iterable

from kvplace.hints import ReuseHint
from kvplace.sim import SHARED, BlockKey, LRUTier, PolicyConfig, ReuseAwareTier
from kvplace.trace import Session

HUMAN = "human"


class SessionLRUTier(ReuseAwareTier):
    """Evict the session idle longest; final sessions first, shared last."""

    def idle(self, sid: str, now: float) -> float:
        # last_use can be a little ahead of `now` (a request's finish time).
        return max(0.0, now - self.last_use.get(sid, now))

    def _score(self, sid: str, now: float):
        if sid == SHARED:
            return (-math.inf, 0.0)
        if sid in self.final:
            return (math.inf, 0.0)
        return (0.0, self.idle(sid, now))


class _ToolTracking:
    """Mixin: remember the tool named by each session's latest hint."""

    def set_hint(self, session_id: str, hint: ReuseHint | None, now: float) -> None:
        stale = hint is not None and hint.turn < self.latest_turn.get(session_id, -1)
        super().set_hint(session_id, hint, now)
        if not stale:
            # The turn's own (identity) hint has no tool: the session is busy.
            self.tool[session_id] = None if hint is None else hint.tool


class OneBitSessionTier(_ToolTracking, SessionLRUTier):
    """Sessions waiting on a human are evicted first; session LRU within
    each class."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.tool: dict[str, str | None] = {}

    def _score(self, sid: str, now: float):
        cls, idle = super()._score(sid, now)
        if math.isinf(cls):
            return (cls, idle)
        return (1.0 if self.tool.get(sid) == HUMAN else 0.0, idle)


class _FirstThenLRU(SessionLRUTier):
    """Evict whole sessions for which `_first` holds (final ones, then the
    longest idle), tail blocks first; then blocks in the `lru` tier's order,
    the inserting request's own sessions last."""

    def _first(self, sid: str, now: float) -> bool:
        raise NotImplementedError

    def _evict(self, n: int, now: float, protect: set[str]) -> list[BlockKey]:
        first = [
            s
            for s, idxs in self.by_session.items()
            if idxs
            and s not in protect
            and s != SHARED
            and (s in self.final or self._first(s, now))
        ]
        first.sort(key=lambda s: SessionLRUTier._score(self, s, now), reverse=True)
        evicted: list[BlockKey] = []
        for sid in first:
            for i in sorted(self.by_session[sid], reverse=True)[: n - len(evicted)]:
                self._drop((sid, i))
                evicted.append((sid, i))
            if len(evicted) >= n:
                return evicted
        for allow_protected in (False, True):
            todo = []
            for k in self.blocks:
                if len(evicted) + len(todo) >= n:
                    break
                if allow_protected or k[0] not in protect:
                    todo.append(k)
            for k in todo:
                self._drop(k)
            evicted += todo
        return evicted


class OneBitTier(_ToolTracking, _FirstThenLRU):
    """Sessions waiting on a human first (oldest first), then block LRU."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.tool: dict[str, str | None] = {}

    def _first(self, sid: str, now: float) -> bool:
        return self.tool.get(sid) == HUMAN


class TTLTier(_FirstThenLRU):
    """Sessions idle longer than `ttl_s` first (oldest first), then block
    LRU."""

    def __init__(self, capacity: int, ttl_s: float):
        super().__init__(capacity)
        self.ttl_s = ttl_s

    def _first(self, sid: str, now: float) -> bool:
        return self.idle(sid, now) > self.ttl_s


class GapDistribution:
    """Online empirical distribution of gaps (all observations, sorted)."""

    def __init__(self):
        self.xs: list[float] = []

    def add(self, gap_s: float) -> None:
        bisect.insort(self.xs, gap_s)

    def __len__(self) -> int:
        return len(self.xs)

    def remaining(self, elapsed: float, min_tail: int = 1) -> float | None:
        """Median of (gap - elapsed) over observed gaps longer than
        `elapsed`; None if fewer than `min_tail` such gaps."""
        i = bisect.bisect_right(self.xs, elapsed)
        m = len(self.xs) - i
        if m < max(1, min_tail):
            return None
        return self.xs[i + m // 2] - elapsed


class HazardTier(SessionLRUTier):
    """Evict the session with the largest expected remaining idle time given
    how long it has been idle, from all gaps observed so far. Past the
    longest observed gap (or before any), the remaining time is taken to be
    the elapsed time (Lindy), which keeps the order LRU-like there."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.gaps = GapDistribution()

    def observe(self, tool_name: str | None, duration_s: float) -> None:
        self.gaps.add(duration_s)

    def remaining(self, sid: str, idle: float) -> float:
        r = self.gaps.remaining(idle)
        return idle if r is None else r

    def _score(self, sid: str, now: float):
        cls, idle = super()._score(sid, now)
        if math.isinf(cls):
            return (cls, idle)
        return (0.0, self.remaining(sid, idle))


class ToolHazardTier(_ToolTracking, HazardTier):
    """`HazardTier` conditioned on the tool the session is running, falling
    back to all gaps when that tool has fewer than `min_tail` observed gaps
    longer than the elapsed time."""

    def __init__(self, capacity: int, min_tail: int = 3):
        super().__init__(capacity)
        self.min_tail = min_tail
        self.tool: dict[str, str | None] = {}
        self.by_tool: dict[str, GapDistribution] = defaultdict(GapDistribution)

    def observe(self, tool_name: str | None, duration_s: float) -> None:
        super().observe(tool_name, duration_s)
        if tool_name is not None:
            self.by_tool[tool_name].add(duration_s)

    def remaining(self, sid: str, idle: float) -> float:
        tool = self.tool.get(sid)
        if tool is not None and tool in self.by_tool:
            r = self.by_tool[tool].remaining(idle, self.min_tail)
            if r is not None:
                return r
        return super().remaining(sid, idle)


class NoAgingTier(ReuseAwareTier):
    """`ReuseAwareTier` whose overdue hinted sessions keep their deadline as
    score (so they look about to return). With `age_unhinted=False`,
    unhinted sessions also keep last_use + default horizon (the pre-fix
    behaviour described in DIRECTION.md)."""

    def __init__(self, capacity: int, age_unhinted: bool = True):
        super().__init__(capacity)
        self.age_unhinted = age_unhinted

    def _score(self, sid: str, now: float) -> float:
        if sid == SHARED or sid in self.final:
            return super()._score(sid, now)
        t = self.deadline.get(sid)
        if t is not None:
            return t
        if self.age_unhinted:
            return super()._score(sid, now)
        return self.last_use.get(sid, now) + self.default_horizon_s


class OracleTier(ReuseAwareTier):
    """`ReuseAwareTier` scored with the trace's true gap and session end:
    any hint for turn k (including the turn's own identity hint at its
    finish) is replaced by the truth for turn k. Later hints for the same
    turn only refresh last use, so the deadline stays exact. Assumes
    time_scale == 1."""

    def __init__(self, capacity: int, sessions: dict[str, Session]):
        super().__init__(capacity)
        self.sessions = sessions

    def set_hint(self, session_id: str, hint: ReuseHint | None, now: float) -> None:
        s = self.sessions.get(session_id)
        if hint is None or s is None or not 0 <= hint.turn < len(s.turns):
            super().set_hint(session_id, hint, now)
            return
        known = session_id in self.deadline or session_id in self.final
        if known and hint.turn == self.latest_turn.get(session_id):
            self.last_use[session_id] = now
            return
        final = hint.turn == len(s.turns) - 1
        gap = None if final else s.turns[hint.turn].tool_duration_s
        super().set_hint(session_id, ReuseHint(session_id, hint.turn, gap, final), now)


def make_tier(kind: str, capacity: int, sessions: dict[str, Session]) -> LRUTier:
    name, _, arg = kind.partition(":")
    if name == "slru":
        return SessionLRUTier(capacity)
    if name == "onebit":
        return OneBitTier(capacity)
    if name == "onebit_slru":
        return OneBitSessionTier(capacity)
    if name == "ttl":
        return TTLTier(capacity, float(arg))
    if name == "hazard":
        return HazardTier(capacity)
    if name == "tool_hazard":
        return ToolHazardTier(capacity)
    if name == "noaging":
        return NoAgingTier(capacity)
    if name == "noaging_all":
        return NoAgingTier(capacity, age_unhinted=False)
    if name == "oracle":
        return OracleTier(capacity, sessions)
    raise ValueError(f"unknown tier policy {kind!r}")


def _cpu_only(name: str, cpu_policy: str) -> PolicyConfig:
    # Same GPU tier as gpu_cpu_aware; only the CPU->disk rule changes.
    return PolicyConfig(name, cpu_policy, gpu_policy="reuse")


BASELINE_POLICIES: dict[str, PolicyConfig] = {
    "slru": _cpu_only("slru", "slru"),
    "onebit": _cpu_only("onebit", "onebit"),
    "onebit_slru": _cpu_only("onebit_slru", "onebit_slru"),
    "hazard": _cpu_only("hazard", "hazard"),
    "tool_hazard": _cpu_only("tool_hazard", "tool_hazard"),
    "ours_noaging": _cpu_only("ours_noaging", "noaging"),
    "ours_noaging_all": _cpu_only("ours_noaging_all", "noaging_all"),
    "cpu_oracle": _cpu_only("cpu_oracle", "oracle"),
}


def get_baseline_policy(name: str) -> PolicyConfig:
    """A name from BASELINE_POLICIES, or `ttl:<seconds>`."""
    kind, _, arg = name.partition(":")
    if kind == "ttl":
        return _cpu_only(name, f"ttl:{float(arg):g}")
    if name in BASELINE_POLICIES:
        return BASELINE_POLICIES[name]
    raise KeyError(name)


def eviction_order(tier: ReuseAwareTier, now: float) -> list[str]:
    """Sessions in the order `tier` would evict them (for tests/inspection)."""
    sessions: Iterable[str] = (s for s, idxs in tier.by_session.items() if idxs)
    return sorted(sessions, key=lambda s: tier._score(s, now), reverse=True)
