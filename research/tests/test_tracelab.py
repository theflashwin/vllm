"""TraceLab -> kvplace conversion: token deltas, gaps, compaction splits."""

from kvplace.tracelab import to_sessions


def row(k, inp, out, start, end, human=False, tool=None):
    return {
        "k": k,
        "in": inp,
        "out": out,
        "start": start,
        "end": end,
        "human": human,
        "tool": tool,
    }


def convert(rows, **kw):
    args = dict(token_scale=1.0, max_context=10**9, max_turns=10**9, min_turns=1)
    args.update(kw)
    return to_sessions({"s": rows}, **args)


def test_new_input_excludes_previous_context_and_output():
    (s,) = convert(
        [
            row(0, 1000, 50, 0.0, 2.0),
            row(1, 1200, 20, 5.0, 6.0, tool="Bash"),
            row(2, 1300, 10, 6.5, 7.0, human=True),
        ]
    )
    assert [t.new_input_tokens for t in s.turns] == [1000, 150, 80]
    assert [t.tool_name for t in s.turns] == ["Bash", "human", None]
    assert [t.tool_duration_s for t in s.turns] == [3.0, 0.5, 0.0]
    assert s.prompt_lengths() == [1000, 1200, 1300]


def test_context_shrink_starts_new_segment():
    sessions = convert(
        [
            row(0, 1000, 50, 0.0, 1.0),
            row(1, 1100, 50, 2.0, 3.0),
            row(2, 400, 10, 4.0, 5.0),  # compacted
            row(3, 500, 10, 6.0, 7.0),
        ]
    )
    assert [s.session_id for s in sessions] == ["s#0", "s#1"]
    assert sessions[1].turns[0].new_input_tokens == 400


def test_max_context_truncates_and_marks_final():
    (s,) = convert(
        [row(0, 1000, 50, 0, 1), row(1, 2000, 50, 2, 3), row(2, 9000, 50, 4, 5)],
        max_context=3000,
    )
    assert len(s.turns) == 2
    assert s.turns[-1].tool_name is None
