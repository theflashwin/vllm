"""Reuse-time aware CachePolicy for vLLM's CPU offload tier.

Load out-of-tree (PYTHONPATH must include research/):

    "kv_connector_extra_config": {
        "eviction_policy": "ReuseAwareCachePolicy",
        "cache_policy_module_path": "kvplace.vllm_policy",
        ...
    }

Eviction prefers chunks whose owning session has the farthest predicted next
use (see `kvplace.sim.ReuseAwareTier` for the same rule in the simulator),
falling back to LRU order within a session and for unhinted chunks. Hints
arrive per request via `kvplace.hints`.

Evictable chunks are indexed per owner (session, shared, or unowned) in lazy
LRU heaps, so `evict` scores owners rather than chunks. The policy runs inside
vLLM's scheduler step; scoring every chunk on each call stalled the scheduler.
"""

import heapq
import math
import os
import time
from collections.abc import Iterable, Sequence

from typing_extensions import override

from kvplace.hints import ReuseHint, parse_hint
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

# Under "vllm" so vLLM's log handler prints it.
logger = init_logger(f"vllm.{__name__}")

_SHARED = object()
_UNOWNED = object()
# Unowned chunks score by store time, highest at the oldest or the newest
# store, so the index holds them in both orders.
_UNOWNED_OLDEST = object()
_UNOWNED_NEWEST = object()
# Set KVPLACE_LOG_HINTS=1 to log every hint the policy receives (use it to
# confirm hint-only requests reach the policy on a real server).
_LOG_HINTS = os.environ.get("KVPLACE_LOG_HINTS") == "1"


class ReuseAwareCachePolicy(LRUCachePolicy):
    default_horizon_s: float = 30.0

    def __init__(self, cache_capacity: int, clock=time.monotonic):
        super().__init__(cache_capacity)
        self._clock = clock
        # Key -> session id, or _SHARED once a second session uses it.
        self._owner: dict[OffloadKey, object] = {}
        self._deadline: dict[str, float] = {}
        self._final: set[str] = set()
        self._last_use: dict[str, float] = {}
        self._latest_turn: dict[str, int] = {}
        # Store time of keys no hinted request has used yet.
        self._stored_at: dict[OffloadKey, float] = {}
        # Index owner -> lazy min-heap of (order, rank, key) over evictable
        # chunks in eviction order. Entries go stale when a chunk is pinned,
        # re-ranked, re-owned or removed; `_is_live` filters them.
        self._by_owner: dict[object, list[tuple[float, int, OffloadKey]]] = {}
        self._entries = 0
        self.hints_received = 0

    def _observe(self, keys: Iterable[OffloadKey], hint: ReuseHint | None) -> None:
        if hint is None:
            return
        sid = hint.session_id
        for key in keys:
            owner = self._owner.get(key)
            if owner is None:
                self._owner[key] = sid
                self._stored_at.pop(key, None)
            elif owner is not _SHARED and owner != sid:
                self._owner[key] = _SHARED
            else:
                continue
            if key in self._evictable:
                self._push_evictable(key)

    def _owner_of(self, key: OffloadKey) -> object:
        return self._owner.get(key, _UNOWNED)

    def _index_owners(self, key: OffloadKey) -> tuple[object, ...]:
        owner = self._owner_of(key)
        if owner is _UNOWNED:
            return (_UNOWNED_OLDEST, _UNOWNED_NEWEST)
        return (owner,)

    def _entry(self, owner: object, key: OffloadKey) -> tuple[float, int, OffloadKey]:
        rank = self._ranks[key]
        # A missing store time scores as "stored now", i.e. newest.
        if owner is _UNOWNED_OLDEST:
            return (self._stored_at.get(key, math.inf), rank, key)
        if owner is _UNOWNED_NEWEST:
            return (-self._stored_at.get(key, math.inf), rank, key)
        return (rank, rank, key)

    def _is_live(self, owner: object, entry: tuple[float, int, OffloadKey]) -> bool:
        _, rank, key = entry
        return (
            key in self._evictable
            and self._ranks.get(key) == rank
            and owner in self._index_owners(key)
        )

    def _score(self, owner: object, key: OffloadKey, now: float) -> float:
        """Predicted next-use time of `key`, the next chunk `owner` would
        evict; higher evicts first."""
        if owner is _SHARED:
            return -math.inf
        if owner is _UNOWNED_OLDEST or owner is _UNOWNED_NEWEST:
            # Unowned: assume reuse a default horizon after storing, aged
            # like an overdue hint.
            t = self._stored_at.get(key, now) + self.default_horizon_s
        elif owner in self._final:
            return math.inf
        elif (t := self._deadline.get(owner)) is None:
            # Unhinted: assume reuse a default horizon after last use, and let
            # it age like an overdue hint so idle sessions do not stick.
            t = self._last_use.get(owner, now) + self.default_horizon_s
        return t if t >= now else now + (now - t)

    @override
    def _push_evictable(self, key: OffloadKey) -> None:
        for owner in self._index_owners(key):
            heap = self._by_owner.setdefault(owner, [])
            heapq.heappush(heap, self._entry(owner, key))
            self._entries += 1

    @override
    def _update_recency(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if self._assign_new_rank(key):
                self._push_evictable(key)
        self._maybe_compact_heap()

    @override
    def _maybe_compact_heap(self) -> None:
        if self._entries < max(64, 4 * len(self._evictable)):
            return
        self._by_owner = {}
        self._entries = 0
        for key in self._evictable:
            for owner in self._index_owners(key):
                self._by_owner.setdefault(owner, []).append(self._entry(owner, key))
                self._entries += 1
        for heap in self._by_owner.values():
            heapq.heapify(heap)

    @override
    def on_store_miss(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        keys = list(keys)
        hint = parse_hint(req_context)
        if hint is None:
            now = self._clock()
            for key in keys:
                self._stored_at.setdefault(key, now)
        self._observe(keys, hint)
        super().on_store_miss(keys, req_context)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        hint = parse_hint(req_context)
        if hint is not None:
            self.hints_received += 1
            if _LOG_HINTS:
                logger.info("kvplace hint %s (req %s)", hint, req_context.req_id)
            sid = hint.session_id
            if hint.turn >= self._latest_turn.get(sid, -1):
                self._latest_turn[sid] = hint.turn
                now = self._clock()
                self._last_use[sid] = now
                self._deadline.pop(sid, None)
                if hint.final:
                    self._final.add(sid)
                else:
                    self._final.discard(sid)
                    if hint.expected_reuse_s is not None:
                        self._deadline[sid] = now + hint.expected_reuse_s
            for keys in key_groups:
                self._observe(keys, hint)
        super().on_request_finished(
            key_groups, insertion_only_keys, reused_keys, req_context
        )

    @override
    def remove(self, key: OffloadKey) -> None:
        super().remove(key)
        self._owner.pop(key, None)
        self._stored_at.pop(key, None)

    @override
    def clear(self) -> None:
        super().clear()
        self._owner.clear()
        self._stored_at.clear()
        self._deadline.clear()
        self._final.clear()
        self._last_use.clear()
        self._latest_turn.clear()
        self._by_owner.clear()
        self._entries = 0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self._clock()
        selected: list[tuple[object, tuple[float, int, OffloadKey]]] = []
        selected_keys: set[OffloadKey] = set()
        # Popped entries to push back: protected keys, and second entries of
        # selected unowned keys (needed if eviction fails).
        deferred: list[tuple[object, tuple[float, int, OffloadKey]]] = []

        def ready(owner: object) -> bool:
            """Pop until the owner's top entry is a live candidate."""
            heap = self._by_owner[owner]
            while heap:
                entry = heap[0]
                if not self._is_live(owner, entry):
                    heapq.heappop(heap)
                    self._entries -= 1
                elif entry[2] in selected_keys or entry[2] in protected:
                    deferred.append((owner, heapq.heappop(heap)))
                    self._entries -= 1
                else:
                    return True
            return False

        # Merge owners by the score of their next chunk: highest predicted
        # next use first, then least recent (within a session, the prefix
        # tail first).
        tops = []
        for i, owner in enumerate(list(self._by_owner)):
            if ready(owner):
                _, rank, key = self._by_owner[owner][0]
                tops.append((-self._score(owner, key, now), rank, i, owner))
            elif not self._by_owner[owner]:
                del self._by_owner[owner]
        heapq.heapify(tops)
        while tops and len(selected) < n:
            top = heapq.heappop(tops)
            _, _, i, owner = top
            # The other unowned order may have taken this top meanwhile.
            if not ready(owner):
                continue
            _, rank, key = self._by_owner[owner][0]
            current = (-self._score(owner, key, now), rank, i, owner)
            if current != top:
                heapq.heappush(tops, current)
                continue
            entry = heapq.heappop(self._by_owner[owner])
            self._entries -= 1
            selected.append((owner, entry))
            selected_keys.add(entry[2])
            if ready(owner):
                _, rank, key = self._by_owner[owner][0]
                heapq.heappush(tops, (-self._score(owner, key, now), rank, i, owner))

        restore = deferred if len(selected) >= n else deferred + selected
        for owner, entry in restore:
            heapq.heappush(self._by_owner.setdefault(owner, []), entry)
            self._entries += 1
        if len(selected) < n:
            return None

        evicted = []
        for _, (_, _, key) in selected:
            chunk = self.chunks.pop(key)
            assert chunk.ref_cnt == 0
            self._evictable.remove(key)
            del self._ranks[key]
            self._owner.pop(key, None)
            self._stored_at.pop(key, None)
            evicted.append((key, chunk))
        self._maybe_compact_heap()
        return evicted
