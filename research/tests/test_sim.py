"""Sanity checks for the trace generator and offline simulator."""

from kvplace.gen_synthetic import DEFAULT_TOOLS, generate
from kvplace.hints import ReuseHint
from kvplace.prefetch_experiment import PrefetchConfig, PrefetchExperiment
from kvplace.sim import POLICIES, CostModel, PolicyConfig, Simulator, get_policy
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
