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

Week-2 skeleton: `evict` is O(evictable) per call. Week 3 replaces it with a
per-session index.
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

logger = init_logger(__name__)

_SHARED = object()
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
        self.hints_received = 0

    def _observe(self, keys: Iterable[OffloadKey], hint: ReuseHint | None) -> None:
        if hint is None:
            return
        sid = hint.session_id
        for key in keys:
            owner = self._owner.get(key)
            if owner is None:
                self._owner[key] = sid
            elif owner is not _SHARED and owner != sid:
                self._owner[key] = _SHARED

    def _score(self, key: OffloadKey, now: float) -> float:
        """Predicted next-use time of the chunk's session; higher evicts first."""
        owner = self._owner.get(key)
        if owner is _SHARED:
            return -math.inf
        if owner is None:
            return now + self.default_horizon_s
        sid = owner
        if sid in self._final:
            return math.inf
        if (t := self._deadline.get(sid)) is not None:
            return t if t >= now else now + (now - t)
        return self._last_use.get(sid, now) + self.default_horizon_s

    @override
    def on_store_miss(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        keys = list(keys)
        self._observe(keys, parse_hint(req_context))
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
            now = self._clock()
            self._last_use[sid] = now
            self._deadline.pop(sid, None)
            if hint.final:
                self._final.add(sid)
            elif hint.expected_reuse_s is not None:
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

    @override
    def clear(self) -> None:
        super().clear()
        self._owner.clear()
        self._deadline.clear()
        self._final.clear()
        self._last_use.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self._clock()
        # Highest predicted next use first; within a session, least recent
        # (the prefix tail) first.
        victims = heapq.nlargest(
            n,
            (k for k in self._evictable if k not in protected),
            key=lambda k: (self._score(k, now), -self._ranks[k]),
        )
        if len(victims) < n:
            return None
        evicted = []
        for key in victims:
            chunk = self.chunks.pop(key)
            assert chunk.ref_cnt == 0
            self._evictable.remove(key)
            del self._ranks[key]
            self._owner.pop(key, None)
            evicted.append((key, chunk))
        self._maybe_compact_heap()
        return evicted
