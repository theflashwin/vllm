"""Sanity checks for the trace generator and offline simulator."""

from kvplace.gen_synthetic import DEFAULT_TOOLS, generate
from kvplace.sim import POLICIES, CostModel, Simulator
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
