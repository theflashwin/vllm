"""LRU CPU-tier policy that also counts promotion waste in tiered offload.

Load out-of-tree (it is imported inside the engine process):

    "eviction_policy": "InstrumentedLRU",
    "cache_policy_module_path": "kvplace.instrument",

Importing this module patches `TieringOffloadingManager` to record, per
chunk, disk->CPU promotions and CPU->GPU loads. The policy records evictions.
A promoted chunk evicted before any load is a wasted promotion read. Totals
are logged every `KVPLACE_STATS_EVERY_S` seconds (default 30) under the
`vllm.kvplace` logger, so they reach vLLM's log handlers.
"""

import logging
import os
import time
from collections import Counter
from collections.abc import Iterable

from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

logger = logging.getLogger("vllm.kvplace.instrument")

_promoted_unloaded: set = set()  # promoted since the last load or eviction
_promotions: Counter = Counter()  # key -> number of promotions
_stats = Counter()
_last_log = [time.monotonic()]
_EVERY = float(os.environ.get("KVPLACE_STATS_EVERY_S", "30"))


def _maybe_log() -> None:
    now = time.monotonic()
    if now - _last_log[0] < _EVERY:
        return
    _last_log[0] = now
    repeats = sum(c - 1 for c in _promotions.values() if c > 1)
    logger.warning(
        "kvplace stats promotions=%d unique_promoted=%d repeat_promotions=%d "
        "loads=%d evictions=%d wasted_promotions=%d",
        _stats["promotions"],
        len(_promotions),
        repeats,
        _stats["loads"],
        _stats["evictions"],
        _stats["wasted"],
    )


def _patch() -> None:
    try:
        from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager
    except ImportError as e:  # pragma: no cover - depends on vLLM version
        logger.warning("kvplace instrument: cannot import tiering manager: %s", e)
        return
    orig_promote = TieringOffloadingManager._initiate_promotion
    orig_load = TieringOffloadingManager.prepare_load

    def _initiate_promotion(self, tier_idx, key, req_context):
        ok = orig_promote(self, tier_idx, key, req_context)
        if ok:
            _stats["promotions"] += 1
            _promotions[key] += 1
            _promoted_unloaded.add(key)
            _maybe_log()
        return ok

    def prepare_load(self, keys, req_context):
        keys = list(keys)
        _stats["loads"] += len(keys)
        _promoted_unloaded.difference_update(keys)
        return orig_load(self, keys, req_context)

    TieringOffloadingManager._initiate_promotion = _initiate_promotion
    TieringOffloadingManager.prepare_load = prepare_load
    logger.warning("kvplace instrument: patched TieringOffloadingManager")


_patch()


class InstrumentedLRU(LRUCachePolicy):
    def evict(self, n: int, protected: set):
        evicted = super().evict(n, protected)
        if evicted:
            keys: Iterable = (k for k, _ in evicted)
            for key in keys:
                _stats["evictions"] += 1
                if key in _promoted_unloaded:
                    _promoted_unloaded.discard(key)
                    _stats["wasted"] += 1
            _maybe_log()
        return evicted
