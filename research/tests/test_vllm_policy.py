"""ReuseAwareCachePolicy driven through vLLM's real CPUOffloadingManager."""

import pytest
from kvplace.hints import ReuseHint

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def keys(*ints: int):
    return [make_offload_key(str(i).encode(), 0) for i in ints]


def ctx(req_id: str, hint: ReuseHint | None) -> ReqContext:
    params = hint.to_kv_transfer_params() if hint else None
    return ReqContext(req_id=req_id, kv_transfer_params=params)


@pytest.fixture
def setup():
    manager = CPUOffloadingManager(
        num_chunks=4,
        cache_policy="ReuseAwareCachePolicy",
        cache_policy_module_path="kvplace.vllm_policy",
        store_threshold=0,
    )
    clock = FakeClock()
    manager._policy._clock = clock
    return manager, clock


def run_request(manager, req_ctx, ks):
    out = manager.prepare_store(ks, req_ctx)
    assert out is not None
    manager.complete_store(ks, req_ctx)
    manager.on_request_finished(req_ctx)
    return out


def test_evicts_session_with_farthest_reuse_not_lru(setup):
    """LRU would evict session a (older); a's next turn is imminent while b's
    tool runs for minutes, so b must go."""
    manager, clock = setup
    run_request(manager, ctx("a0", ReuseHint("a", 0, 1.0)), keys(1, 2))
    clock.t = 0.5
    run_request(manager, ctx("b0", ReuseHint("b", 0, 300.0)), keys(3, 4))

    out = run_request(manager, ctx("c0", ReuseHint("c", 0, 10.0)), keys(5, 6))

    assert set(out.evicted_keys) == set(keys(3, 4))
    for k in keys(1, 2):
        assert manager.lookup(k, ReqContext(req_id="x")) is LookupResult.HIT


def test_final_session_evicted_first(setup):
    """The finished session is the most recent, so LRU would keep it."""
    manager, clock = setup
    run_request(manager, ctx("b0", ReuseHint("b", 0, 300.0)), keys(3, 4))
    run_request(manager, ctx("a0", ReuseHint("a", 0, final=True)), keys(1, 2))

    out = run_request(manager, ctx("c0", ReuseHint("c", 0, 10.0)), keys(5))

    assert out.evicted_keys == keys(2)  # tail of the finished session first


def test_shared_prefix_kept(setup):
    """Chunk 1 is used by two sessions and is the LRU victim; keep it."""
    manager, clock = setup
    run_request(manager, ctx("a0", ReuseHint("a", 0, 100.0)), keys(1, 2))
    run_request(manager, ctx("b0", ReuseHint("b", 0, 100.0)), keys(1, 3))
    manager.touch(keys(2, 3), ReqContext(req_id="t"))
    run_request(manager, ctx("c0", None), keys(4))

    out = run_request(manager, ctx("d0", None), keys(5))

    assert len(out.evicted_keys) == 1
    assert out.evicted_keys[0] in keys(2, 3)


def test_unhinted_requests_fall_back_to_lru(setup):
    manager, clock = setup
    run_request(manager, ctx("r0", None), keys(1, 2))
    run_request(manager, ctx("r1", None), keys(3, 4))

    out = run_request(manager, ctx("r2", None), keys(5, 6))

    assert set(out.evicted_keys) == set(keys(1, 2))


def test_hint_only_request_updates_deadline(setup):
    """A post-response hint evicts original blocks, not only its padding block."""
    manager, clock = setup
    run_request(manager, ctx("a0", ReuseHint("a", 0)), keys(1, 2))
    run_request(manager, ctx("b0", ReuseHint("b", 0, 300.0)), keys(3))
    run_request(manager, ctx("a0-hint", ReuseHint("a", 0, final=True)), keys(4))

    # Keep the padding block in the request so it cannot satisfy eviction.
    out = run_request(manager, ctx("c0", None), keys(4, 5))

    assert out.evicted_keys[0] in keys(1, 2)
    assert manager.lookup(keys(3)[0], ReqContext(req_id="x")) is LookupResult.HIT


def test_late_hint_does_not_replace_a_newer_turn(setup):
    """A delayed hint for turn 0 must not extend turn 1's reuse deadline."""
    manager, _ = setup
    run_request(manager, ctx("a0", ReuseHint("a", 0)), keys(1))
    run_request(manager, ctx("a1", ReuseHint("a", 1, 5.0)), keys(2))
    run_request(manager, ctx("b0", ReuseHint("b", 0, 100.0)), keys(3))
    run_request(manager, ctx("a0-hint", ReuseHint("a", 0, 300.0)), keys(4))

    out = run_request(manager, ctx("c0", None), keys(5))

    assert out.evicted_keys == keys(3)


def test_storeless_request_hint_needs_on_new_request(setup):
    """Documents why replay pads hint-only requests: without a store or load,
    the CPU tier only finalizes requests it saw in on_new_request, which the
    tiering manager does not forward to the primary tier."""
    manager, clock = setup
    policy = manager._policy
    hint_ctx = ctx("h", ReuseHint("a", 0, 5.0))
    manager.on_request_finished(hint_ctx)
    assert policy.hints_received == 0

    hint_ctx = ctx("h2", ReuseHint("a", 0, 5.0))
    manager.on_new_request(hint_ctx)  # CPU-only spec does call this
    manager.on_request_finished(hint_ctx)
    assert policy.hints_received == 1
