"""Reuse-time predictors that turn a trace into per-request hints.

Shared by the simulator and the replay client so both see identical hints.

Two timings:
  * submission-time (default): the hint rides on the turn's own request, so
    the predictor may only use what is known before the model responds.
    `oracle`/`noisy` break this rule on purpose — they are upper bounds.
  * post-response (`post_response = True`): the hint is sent in a separate
    hint-only request right after the response, when the agent knows which
    tool it is calling and whether the session is ending.

Predictors with an `observe(tool_name, duration_s)` method are fed every
completed tool call (in time order) so they can learn online.
"""

import math
import random
import statistics
import zlib
from collections import defaultdict, deque
from collections.abc import Callable

from kvplace.hints import ReuseHint
from kvplace.trace import Session

# (session, turn index, observed gaps so far in this session) -> hint
Predictor = Callable[[Session, int, list[float]], ReuseHint | None]


def no_hints(session: Session, k: int, gaps: list[float]) -> ReuseHint | None:
    return None


def oracle(session: Session, k: int, gaps: list[float]) -> ReuseHint:
    turn = session.turns[k]
    final = k == len(session.turns) - 1
    return ReuseHint(
        session.session_id,
        k,
        None if final else turn.tool_duration_s,
        final,
    )


def make_noisy_oracle(sigma: float, seed: int = 0) -> Predictor:
    """Oracle gap times a log-normal error with log-space std `sigma`.
    The final-turn flag stays exact."""

    def predict(session: Session, k: int, gaps: list[float]) -> ReuseHint:
        h = oracle(session, k, gaps)
        if h.expected_reuse_s is None:
            return h
        # Stable across processes, unlike hash() on str.
        rng = random.Random(zlib.crc32(f"{seed}:{session.session_id}:{k}".encode()))
        err = math.exp(rng.gauss(0.0, sigma))
        return ReuseHint(h.session_id, k, h.expected_reuse_s * err, h.final)

    return predict


def make_session_ewma(alpha: float = 0.5) -> Predictor:
    """EWMA of this session's previous gaps; no knowledge of session end."""

    def predict(session: Session, k: int, gaps: list[float]) -> ReuseHint:
        if not gaps:
            return ReuseHint(session.session_id, k, None)
        est = gaps[0]
        for g in gaps[1:]:
            est = alpha * g + (1 - alpha) * est
        return ReuseHint(session.session_id, k, est)

    return predict


class ToolMedian:
    """Post-response: median of recently observed durations of the tool the
    agent just chose (learned online across all sessions). The final flag is
    exact, since a response without a tool call ends the session."""

    post_response = True

    def __init__(self, window: int = 200, min_samples: int = 3):
        self.min_samples = min_samples
        self.history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def observe(self, tool_name: str | None, duration_s: float) -> None:
        if tool_name is not None:
            self.history[tool_name].append(duration_s)

    def __call__(self, session: Session, k: int, gaps: list[float]) -> ReuseHint:
        turn = session.turns[k]
        if k == len(session.turns) - 1:
            return ReuseHint(session.session_id, k, None, final=True)
        h = self.history.get(turn.tool_name)
        est = statistics.median(h) if h and len(h) >= self.min_samples else None
        return ReuseHint(session.session_id, k, est)


def is_post_response(predictor: Predictor) -> bool:
    return getattr(predictor, "post_response", False)


def get_predictor(name: str) -> Predictor:
    """Names: none, oracle, noisy:<sigma>, ewma[:<alpha>], tool[:<window>]."""
    kind, _, arg = name.partition(":")
    if kind == "none":
        return no_hints
    if kind == "oracle":
        return oracle
    if kind == "noisy":
        return make_noisy_oracle(float(arg or 1.0))
    if kind == "ewma":
        return make_session_ewma(float(arg or 0.5))
    if kind == "tool":
        return ToolMedian(int(arg or 200))
    raise ValueError(f"unknown predictor {name!r}")
