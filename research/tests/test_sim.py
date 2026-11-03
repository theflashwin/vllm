"""Sanity checks for the trace generator and offline simulator."""

from kvplace.gen_synthetic import DEFAULT_TOOLS, generate
from kvplace.hints import ReuseHint
from kvplace.prefetch_experiment import PrefetchConfig, PrefetchExperiment
from kvplace.sim import (
    POLICIES,
    ContextPolicy,
    CostModel,
    PolicyConfig,
    Simulator,
    _block_keys,
    chain_position,
    get_policy,
)
from kvplace.trace import Session, Turn, read_trace, write_trace


def small_trace(n: int = 40, seed: int = 0):
    return generate(
        n, 0.5, 8, 30, (500, 1500), (50, 200), 512, 20000, DEFAULT_TOOLS, seed
    )


def run(sessions, policy, predictor="oracle", gpu=200, cpu=800, sec=10**7):
    return Simulator(
        sessions, POLICIES[policy], predictor, CostModel(), gpu, cpu, sec
    ).run()


def test_trace_roundtrip(tmp_path):
    sessions = small_trace(5)
    write_trace(sessions, tmp_path / "t.jsonl")
    assert read_trace(tmp_path / "t.jsonl") == sessions


def test_prompts_grow_by_previous_output():
    s = Session("s", 0.0, [Turn(100, 10, "grep", 1.0), Turn(20, 5)])
    assert s.prompt_lengths() == [100, 130]


def test_every_turn_replayed_once():
    sessions = small_trace()
    result = run(sessions, "lru")
    assert len(result.records) == sum(len(s.turns) for s in sessions)


def test_unbounded_tiers_lose_nothing():
    """With huge GPU capacity every reusable block is a GPU hit."""
    result = run(small_trace(), "lru", gpu=10**7)
    s = result.summary()
    assert s["resume_frac_gpu"] == 1.0


def test_oracle_reuse_policy_beats_lru_under_pressure():
    sessions = small_trace(80)
    lru = run(sessions, "lru").summary()
    reuse = run(sessions, "reuse").summary()
    assert reuse["resume_ttft_mean"] < lru["resume_ttft_mean"]
    assert reuse["resume_frac_sec"] < lru["resume_frac_sec"]


def test_tool_predictor_adds_the_same_hint_request_load_to_baseline():
    sessions = small_trace(5)
    expected = sum(len(s.turns) for s in sessions)
    lru = run(sessions, "lru", predictor="tool")
    reuse = run(sessions, "reuse", predictor="tool")

    assert lru.counters.hint_requests == expected
    assert reuse.counters.hint_requests == expected
    assert (
        lru.counters.gpu_to_cpu_store_blocks
        > run(sessions, "lru", predictor="none").counters.gpu_to_cpu_store_blocks
    )


def prefetch_trace():
    return [
        Session(str(i), i * 4.0, [Turn(256, 0, "grep", 1.0), Turn(16, 0)])
        for i in range(12)
    ]


def test_prefetch_budgets_bound_resident_and_inflight_blocks():
    """Transferred bytes must be accounted for, including discarded completions."""
    config = PrefetchConfig(chunk_blocks=4, capacity_fraction=0.5)
    sim = PrefetchExperiment(prefetch_trace(), "budget", CostModel(), 0, 8, config)
    result = sim.run()
    c = result.counters

    assert c.sec_to_cpu_prefetch_blocks > 0
    assert sim.metrics["peak_prefetch_blocks"] <= 4
    assert (
        c.prefetch_used_blocks + c.prefetch_wasted_blocks
        == c.sec_to_cpu_prefetch_blocks
    )
    assert c.hint_requests == 24
    end = max(r.arrival_s + r.ttft_s for r in result.records)
    rate = config.bandwidth_fraction * sim.cost.sec_cpu_gbps * 1e9
    assert c.sec_to_cpu_prefetch_blocks * sim.cost.block_bytes <= (
        config.chunk_blocks * sim.cost.block_bytes + rate * end
    )


def test_uncertainty_gate_rejects_calls_outside_residency_window():
    """A median estimate alone must not bypass the probability gate."""
    config = PrefetchConfig(
        chunk_blocks=4, capacity_fraction=0.5, residency_window_s=0.001
    )
    sim = PrefetchExperiment(prefetch_trace(), "uncertain", CostModel(), 0, 8, config)
    result = sim.run()

    assert sim.metrics["prefetch_rejected_uncertain"] > 0
    assert sim.metrics["prefetch_rejected_cold"] > 0
    assert result.counters.sec_to_cpu_prefetch_blocks == 0


def test_prefetch_completion_after_resume_is_wasted_and_demand_waits():
    """In-flight reads occupy bandwidth without prematurely populating CPU."""
    sessions = prefetch_trace()
    cost = CostModel(sec_cpu_gbps=0.01)
    sim = PrefetchExperiment(sessions, "fixed", cost, 0, 8, PrefetchConfig())
    result = sim.run()

    assert sim.metrics["prefetch_stale_blocks"] > 0
    assert sim.metrics["demand_link_wait_s"] > 0
    c = result.counters
    assert (
        c.prefetch_used_blocks + c.prefetch_wasted_blocks
        == c.sec_to_cpu_prefetch_blocks
    )


def _evicted_during_gap(tokens: int) -> list[Session]:
    """Session a pauses 60 s; session b fills the GPU meanwhile."""
    return [
        Session("a", 0.0, [Turn(tokens, 16, "build", 60.0), Turn(16, 16)]),
        Session("b", 20.0, [Turn(tokens, 16)]),
    ]


def _warm(sessions, policy, predictor, gpu=0):
    return Simulator(
        sessions, get_policy(policy), predictor, CostModel(), gpu, 10**6, 10**7
    ).run()


def test_oracle_warmup_restores_before_return():
    """A warm-up timed by the oracle lands the prefix on GPU before the
    session returns, so the resumption is a GPU hit, not a CPU load."""
    s = _evicted_during_gap(4096)
    lru = _warm(s, "lru", "none", gpu=300).records
    timed = _warm(s, "warm:5", "oracle", gpu=300).records
    resume = [r for r in timed if r.session_id == "a" and r.turn == 1][0]
    assert [r for r in lru if r.session_id == "a"][1].gpu_hit_blocks < 4096 // 16
    assert resume.gpu_hit_blocks == 4096 // 16


def test_warmup_sent_too_late_is_waited_for():
    """A lead shorter than the restore time makes the request wait for the
    in-flight warm-up instead of seeing an instant GPU hit."""
    s = _evicted_during_gap(160_000)
    result = _warm(s, "warm:0", "oracle", gpu=10_100)
    resume = [r for r in result.records if r.session_id == "a" and r.turn == 1][0]
    assert result.counters.warmup_late == 1
    assert resume.ttft_s > CostModel().cpu_to_gpu_s(10_000)


def test_overestimated_gap_cancels_the_warmup():
    """If the session returns before the scheduled warm-up, it is cancelled
    rather than restoring stale state."""
    s = [Session("s", 0.0, [Turn(1600, 16, "x", 1.0), Turn(16, 16)])]
    result = Simulator(
        s, PolicyConfig("w", warmup_lead_s=0.0), "oracle", CostModel(), 0, 10**6, 0
    )
    # Force a 100 s prediction for a 1 s gap.
    result.predictor = lambda sess, k, gaps: ReuseHint("s", k, 100.0, k == 1)
    counters = result.run().counters
    assert counters.warmup_cancelled == 1
    assert counters.hint_requests == 1  # only the final turn's warm-up


def test_contended_secondary_link_serializes_reads():
    """With a shared secondary link, a read issued while another is in flight
    finishes only after it, so prefetch traffic can delay demand loads."""
    sim = Simulator([], POLICIES["lru"], "none", CostModel(), 0, 0, 0, contend_sec=True)
    one = CostModel().sec_to_cpu_s(1000)
    assert sim._secondary_load_s(0.0, 1000) == one
    assert sim._secondary_load_s(0.0, 1000) == 2 * one


def test_gpu_aware_policy_keeps_soon_returning_session_on_gpu():
    """Agent-aware GPU eviction keeps the session that returns soonest, where
    LRU would evict it for being least recently used."""
    sessions = [
        Session("soon", 0.0, [Turn(1600, 16, "x", 30.0), Turn(16, 16)]),
        Session("late", 5.0, [Turn(1600, 16, "x", 600.0), Turn(16, 16)]),
        Session("new", 20.0, [Turn(1600, 16)]),
    ]

    def soon_gpu_hits(policy):
        result = Simulator(
            sessions, POLICIES[policy], "oracle", CostModel(), 210, 10**4, 0
        ).run()
        (r,) = [r for r in result.records if r.session_id == "soon" and r.turn == 1]
        return r.gpu_hit_blocks

    assert soon_gpu_hits("gpu_aware") > soon_gpu_hits("lru")


def test_session_tool_predictor_uses_this_sessions_history():
    """A session that ran a slow test suite before is predicted slow again,
    even when the global median for that tool is short."""
    from kvplace.predictors import get_predictor

    p = get_predictor("toolsess")
    for _ in range(5):
        p.observe("Bash", 1.0)
    s = Session(
        "s", 0.0, [Turn(10, 1, "Bash", 300.0), Turn(10, 1, "Bash", 280.0), Turn(1, 1)]
    )
    assert p(s, 1, []).expected_reuse_s == 300.0


def test_nf_predictor_does_not_know_where_the_session_ends():
    """Without the trace's end-of-session knowledge, a response that calls no
    tool is predicted like a wait for the user, not flagged final."""
    from kvplace.predictors import get_predictor

    s = Session("s", 0.0, [Turn(10, 1, "human", 60.0), Turn(10, 1)])
    exact, nf = get_predictor("tool"), get_predictor("tool-nf")
    for p in (exact, nf):
        for _ in range(3):
            p.observe("human", 60.0)
    assert exact(s, 1, []).final
    hint = nf(s, 1, [])
    assert not hint.final and hint.expected_reuse_s == 60.0


def test_unhinted_idle_session_eventually_evicted_before_hinted_one():
    """A session that never got a prediction must not look "about to return"
    forever: once idle well past the default horizon it is evicted before a
    session predicted to return soon."""
    from kvplace.sim import ReuseAwareTier

    tier = ReuseAwareTier(capacity=4)
    tier.insert([("stale", 0), ("stale", 1)], now=0.0)
    tier.insert([("soon", 0), ("soon", 1)], now=0.0)
    tier.set_hint("soon", ReuseHint("soon", 0, 10.0), now=990.0)
    evicted = tier.insert([("new", 0), ("new", 1)], now=1000.0)
    assert {sid for sid, _ in evicted} == {"stale"}


def test_residual_predictor_switches_to_the_slow_mode():
    """A bimodal tool: the median is the fast mode, but a call that outlives
    it is predicted from the slow calls only."""
    from kvplace.predictors import ToolResidual

    p = ToolResidual(exact_final=False)
    for d in [0.1] * 11 + [100.0] * 9:
        p.observe("Bash", d)
    s = Session("s", 0.0, [Turn(10, 1, "Bash", 0.1), Turn(10, 1)])
    hint = p(s, 0, [])
    assert hint.expected_reuse_s == 0.1
    assert dict(hint.residual)[0.3] == 99.7


def test_late_session_ranked_by_its_residual_wait():
    """Both sessions were predicted back in 1 s and are equally late. The
    residual table says x is now in its slow mode, so x goes first."""
    from kvplace.sim import ReuseAwareTier

    tier = ReuseAwareTier(capacity=4)
    tier.insert([("x", 0), ("x", 1)], now=0.0)
    tier.insert([("y", 0), ("y", 1)], now=0.0)
    tier.set_hint("y", ReuseHint("y", 0, 1.0), now=0.0)
    tier.set_hint("x", ReuseHint("x", 0, 1.0, residual=((0, 1), (1, 600))), now=0.0)
    evicted = tier.insert([("new", 0), ("new", 1)], now=10.0)
    assert {sid for sid, _ in evicted} == {"x"}


def test_expected_remaining_interpolates_and_extends():
    from kvplace.hints import expected_remaining

    table = ((0.0, 1.0), (10.0, 21.0))
    assert expected_remaining(table, 5.0) == 11.0
    assert expected_remaining(table, 50.0) == 50.0


def test_gpu_admission_queues_requests_that_do_not_fit():
    """Two sessions arrive together; each context needs most of the GPU, so
    with admission the second waits for the first to finish decoding."""
    sessions = [
        Session(f"s{i}", 0.0, [Turn(1600, 100), Turn(16, 1)], seed=i) for i in range(2)
    ]

    def first_ttfts(admission):
        r = Simulator(
            sessions, POLICIES["lru"], "none", CostModel(), 150, 0, 10**6,
            gpu_admission=admission,
        ).run()  # fmt: skip
        return sorted(x.ttft_s for x in r.records if x.turn == 0)

    fast, slow = first_ttfts(True)
    assert slow - fast > 100 * CostModel().decode_s_per_tok
    assert first_ttfts(False)[1] < slow


def test_return_probability_conditions_on_time_waited():
    from kvplace.hints import return_probability

    qs = (0.1, 0.1, 0.1, 0.1, 100.0, 100.0)  # mostly instant, some 100 s
    assert return_probability(qs, 0.0, 1.0) > 0.6
    assert return_probability(qs, 5.0, 10.0) == 0.0  # in the slow mode
    assert return_probability(qs, 5.0, 100.0) == 1.0
    assert return_probability(qs, 200.0, 10.0) == 0.0  # outlasted them all


def test_recency_guard_keeps_a_session_that_just_finished():
    """Session x just finished a turn and is predicted to wait long; without the
    guard it goes first, with it the older session y goes instead."""
    from kvplace.sim import ReuseAwareTier

    def evicted(guard):
        tier = ReuseAwareTier(capacity=4, recency_guard_s=guard)
        tier.insert([("y", 0), ("y", 1)], now=0.0)
        tier.set_hint("y", ReuseHint("y", 0, 50.0), now=0.0)
        tier.insert([("x", 0), ("x", 1)], now=9.0)
        tier.set_hint("x", ReuseHint("x", 0, 300.0), now=9.0)
        return {sid for sid, _ in tier.insert([("n", 0), ("n", 1)], now=10.0)}

    assert evicted(0.0) == {"x"}
    assert evicted(5.0) == {"y"}


def _admission_sim(sessions, gpu, **kw):
    return Simulator(
        sessions, POLICIES["lru"], "none", CostModel(), gpu, 0, 10**6,
        gpu_admission=True, **kw,
    )  # fmt: skip


def test_admission_is_fifo_and_lookup_happens_at_admission():
    """The third request fits but arrived after one that does not: FIFO holds
    it back. A session's blocks cached by an earlier finish are seen as hits
    only when it is admitted."""
    big = [Turn(1600, 100), Turn(16, 1)]
    sessions = [
        Session("a", 0.0, big, seed=0),
        Session("b", 0.001, big, seed=1),
        Session("c", 0.002, [Turn(160, 1), Turn(16, 1)], seed=2),
    ]
    recs = {
        r.session_id: r
        for r in _admission_sim(sessions, 150).run().records
        if r.turn == 0
    }
    assert recs["b"].queue_s > 0
    assert recs["c"].queue_s >= recs["b"].queue_s - 0.002


def test_running_blocks_are_pinned_in_one_memory_pool():
    """While a's blocks are pinned, b cannot evict them; the cache gets only
    what running requests leave."""
    sim = _admission_sim([Session("a", 0.0, [Turn(800, 10), Turn(16, 1)], seed=0)], 60)
    sim._hold(10.0, [("a", i) for i in range(50)], 5)
    assert sim.gpu.capacity == 55
    sim.gpu.insert([("a", i) for i in range(50)], 0.0)
    sim.gpu.insert([("b", i) for i in range(20)], 0.0)
    assert all(("a", i) in sim.gpu for i in range(50))
    assert len(sim.gpu) <= 55


def test_small_context_shrink_is_not_compaction():
    from kvplace.tracelab import to_sessions

    row = lambda k, i, o: {  # noqa: E731
        "k": k, "in": i, "out": o, "hit": 0, "start": float(k), "end": k + 0.5,
        "human": False, "tool": "Bash",
    }  # fmt: skip
    rows = {
        "s": [
            row(0, 1000, 50),
            row(1, 1000, 50),
            row(2, 1200, 50),
            row(3, 100, 5),
            row(4, 200, 5),
        ]
    }
    out = to_sessions(rows, 1.0, 10**9, 10**9, 2)
    assert [len(s.turns) for s in out] == [3, 2]
    assert out[0].prompt_lengths()[2] == 1200  # catches up to the recorded context


SCAN = CostModel(
    prefill_tok_per_s=1e12,
    decode_s_per_tok=0.01,
    decode_s_per_ctx_tok=1e-5,
    sched_overhead_s=0,
)


def _next_arrival(sessions, sid="s0", **kw):
    records = Simulator(
        sessions, POLICIES["lru"], "none", SCAN, 10**5, 0, 0, gpu_admission=True, **kw
    ).run().records  # fmt: skip
    return next(r for r in records if r.session_id == sid and r.turn == 1)


def test_decode_step_grows_with_all_decoding_context():
    """Decode time follows the total KV read each step, so a concurrent request
    slows this one until it leaves."""
    alone = [Session("s0", 0.0, [Turn(1600, 100), Turn(16, 1)])]
    assert abs(_next_arrival(alone).arrival_s - 100 * 0.026) < 1e-6

    other = Session("s1", 0.0, [Turn(1600, 50)], seed=1)
    # Together for 50 tokens at 0.042 s, then s0 alone for 50 at 0.026 s.
    assert abs(_next_arrival(alone + [other]).arrival_s - 3.4) < 1e-6


def test_prefill_time_grows_with_the_context_before_it():
    cost = CostModel(prefill_tok_per_s=1e4, prefill_s_per_tok_per_ctx=1e-9)
    assert abs(cost.prefill_s(0, 1000) - (0.1 + 0.0005)) < 1e-9
    assert abs(cost.prefill_s(80_000, 1000) - (0.1 + 0.0805)) < 1e-9


def test_context_limit_drops_tokens_and_loses_the_cache():
    turns = [Turn(1600, 16), Turn(16, 1)]
    keep = _next_arrival([Session("s0", 0.0, turns)])
    assert keep.prompt_tokens == 1632 and keep.miss_blocks == 2

    limit = ContextPolicy(limit=1000, keep=0.5)
    dropped = _next_arrival([Session("s0", 0.0, turns)], context_policy=limit)
    assert dropped.prompt_tokens == 500 + 16 + 16
    assert dropped.miss_blocks == 2


def test_pressure_limit_waits_for_a_queue():
    def run_small(wait):
        sessions = [
            Session(f"s{i}", 0.0, [Turn(1600, 100), Turn(16, 1)], seed=i)
            for i in range(2)
        ]
        limit = ContextPolicy(limit=1000, pressure_wait_s=wait)
        return Simulator(
            sessions, POLICIES["lru"], "none", SCAN, 150, 0, 0,
            gpu_admission=True, context_policy=limit,
        ).run().link.get("context_drops", 0)  # fmt: skip

    assert run_small(10**6) == 0
    assert run_small(0.0) > 0


def test_context_drop_adds_the_turns_to_fetch_it_back():
    limit = ContextPolicy(limit=1000, keep=0.5, reread_turns=2, reread_tokens=100)
    session = Session("s0", 0.0, [Turn(1600, 16), Turn(16, 1)])
    result = Simulator(
        [session], POLICIES["lru"], "none", SCAN, 10**5, 0, 0,
        gpu_admission=True, context_policy=limit,
    ).run()  # fmt: skip
    assert [r.prompt_tokens for r in result.records] == [500, 616, 816, 932]


def test_rereads_that_cross_the_limit_do_not_add_more_rereads():
    limit = ContextPolicy(limit=1000, keep=0.5, reread_turns=3, reread_tokens=300)
    session = Session("s0", 0.0, [Turn(1600, 16)])
    result = Simulator(
        [session], POLICIES["lru"], "none", SCAN, 10**5, 0, 0,
        gpu_admission=True, context_policy=limit,
    ).run()  # fmt: skip
    assert [r.prompt_tokens for r in result.records] == [500, 816, 500, 900]


def _selective(select, turns, **kw):
    limit = ContextPolicy(1000, 0.5, select=select, min_clear=1, **kw)
    session = Session("s0", 0.0, turns)
    result = Simulator(
        [session], POLICIES["lru"], "none", SCAN, 10**5, 0, 0,
        gpu_admission=True, context_policy=limit,
    ).run()  # fmt: skip
    return [r.prompt_tokens for r in result.records]


def _mixed_turns():
    return [
        Turn(100, 0, "Read"),
        Turn(400, 0, "Bash"),
        Turn(400, 0, "Edit"),
        Turn(16, 0, "Edit"),
        Turn(200, 1),
    ]


def test_refetch_clears_only_results_of_rerunnable_tools():
    prompts = _selective("refetch", _mixed_turns(), keep_recent=1, stub_tokens=0)
    assert prompts == [100, 500, 900, 916, 716]


def test_rolling_clears_any_old_tool_result():
    prompts = _selective("rolling", _mixed_turns(), keep_recent=1, stub_tokens=0)
    assert prompts == [100, 500, 900, 916, 300]


def test_selective_clear_keeps_cache_before_the_first_cleared_result():
    session = Session("s0", 0.0, _mixed_turns())
    limit = ContextPolicy(
        1000, 0.5, select="rolling", keep_recent=1, stub_tokens=0, min_clear=1
    )
    Simulator(
        [session], POLICIES["lru"], "none", SCAN, 10**5, 0, 0,
        gpu_admission=True, context_policy=limit,
    ).run()  # fmt: skip
    owners = [owner for owner, _ in _block_keys(session, 128, 16)]
    assert owners == ["s0"] * 6 + ["s0#1"] * 2


def _two_sessions_one_human_turn(queue_order):
    # An agent turn queues before a human turn while a long request holds the GPU.
    blocker = Session("a", 0.0, [Turn(800, 2000)])
    agent = Session("b", 1.0, [Turn(500, 1, "Read"), Turn(500, 1)])
    human = Session("c", 1.0, [Turn(500, 1, "human", 2.0), Turn(500, 1)])
    return Simulator(
        [blocker, agent, human], POLICIES["lru"], "none", SCAN, 100, 0, 0,
        gpu_admission=True, queue_order=queue_order,
    ).run()  # fmt: skip


def test_human_first_queue_admits_every_request():
    assert len(_two_sessions_one_human_turn("human").records) == 5


def test_gated_drop_waits_for_a_long_tool_wait():
    limit = ContextPolicy(1000, 0.5, gate_wait_s=5.0, gate_cap=3000)
    turns = [Turn(1600, 0, "Read", 1.0), Turn(100, 0, "Bash", 9.0), Turn(100, 1)]
    result = Simulator(
        [Session("s0", 0.0, turns)], POLICIES["lru"], "none", SCAN, 10**5, 0, 0,
        gpu_admission=True, context_policy=limit,
    ).run()  # fmt: skip
    assert [r.prompt_tokens for r in result.records] == [1600, 1700, 500]


def _agent_blocking_human(queue_order):
    agent = Session("a", 0.0, [Turn(16, 1, "Read"), Turn(800, 5000)])
    human = Session("h", 5.0, [Turn(800, 1)])
    return Simulator(
        [agent, human], POLICIES["lru"], "none", SCAN, 60, 10**4, 0,
        gpu_admission=True, queue_order=queue_order,
    ).run()  # fmt: skip


def test_preemption_lets_a_human_turn_pass_a_decoding_agent_turn():
    waited = _agent_blocking_human("human")
    preempted = _agent_blocking_human("human_preempt")
    assert preempted.link["preemptions"] == 1
    assert len(preempted.records) == 3

    def ttft(result):
        return next(r.ttft_s for r in result.records if r.session_id == "h")

    assert ttft(preempted) < ttft(waited) / 10


def test_agent_only_limit_skips_human_turns():
    limit = ContextPolicy(1000, 0.5, agent_only=True)
    turns = [Turn(1600, 0, "human", 1.0), Turn(100, 0, "Read", 1.0), Turn(100, 1)]
    result = Simulator(
        [Session("s0", 0.0, turns)], POLICIES["lru"], "none", SCAN, 10**5, 0, 0,
        gpu_admission=True, context_policy=limit,
    ).run()  # fmt: skip
    assert [r.prompt_tokens for r in result.records] == [1600, 1700, 500]


def test_chain_position_counts_from_the_last_human_message():
    turns = [Turn(1, 1, "Read"), Turn(1, 1, "human"), Turn(1, 1, "Bash"), Turn(1, 1)]
    session = Session("s0", 0.0, turns)
    positions = [chain_position(session, k) for k in range(4)]
    assert positions == [0, 1, 0, 1]


def _chain_order(queue_order, **kw):
    blocker = Session("a", 2.0, [Turn(16, 1, "Read"), Turn(800, 1000)])
    old = Session(
        "o", 0.0, [Turn(16, 1, "Read", 1.0), Turn(16, 1, "Read", 2.0), Turn(800, 1)]
    )
    new = Session("n", 4.0, [Turn(800, 1)])
    result = Simulator(
        [blocker, old, new], POLICIES["lru"], "none", SCAN, 60, 10**4, 0,
        gpu_admission=True, queue_order=queue_order, **kw,
    ).run()  # fmt: skip
    big = [r for r in result.records if r.prompt_tokens >= 800 and r.session_id != "a"]
    return [r.session_id for r in big]


def test_chain_las_serves_the_newer_chain_first_and_sticky_the_older():
    assert _chain_order("chain_las")[0] == "n"
    assert _chain_order("chain_sticky")[0] == "o"


def test_chain_aging_lets_a_long_waiting_chain_go_first():
    assert _chain_order("chain_las", chain_aging_s=0.1)[0] == "o"
