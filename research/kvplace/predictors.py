"""Reuse-time predictors that turn a trace into per-request hints.

Shared by the simulator and the replay client so both see identical hints.

Two timings:
  * submission-time (default): the hint rides on the turn's own request, so
    the predictor may only use what is known before the model responds.
    `oracle`/`noisy` break this rule on purpose — they are upper bounds.
  * post-response (`post_response = True`): the turn request carries session
    identity without a reuse estimate. A separate hint-only request updates
    it after the response, when the agent knows which tool it is calling and
    whether the session is ending.

Predictors with an `observe(tool_name, duration_s)` method are fed every
completed tool call (in time order) so they can learn online.
"""

import math
import random
import statistics
import zlib
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import replace

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
        tool=turn.tool_name,
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
        return replace(h, expected_reuse_s=h.expected_reuse_s * err)

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
    """Post-response: median (or `quantile`) of recently observed durations of
    the tool the agent just chose (learned online across all sessions).

    With `exact_final`, the session's last turn is flagged final. That uses
    the trace's knowledge of where the session ends: a real agent only knows
    the response called no tool, i.e. it is waiting on the user. Without it,
    that turn is predicted like any other "human" wait.

    With `end_aware` (and no `exact_final`), the last turn of a segment that
    the converter cut artificially (context compaction, --max-context /
    --max-turns truncation) is predicted with the tool the trace actually
    called next (`session.meta["end_next_tool"]`), which a live agent knows
    once that response finishes. Only genuine session ends, or ends followed
    by a user message, are predicted as "human" waits. Traces without that
    metadata fall back to "human" (same as `end_aware=False`)."""

    post_response = True

    def __init__(
        self,
        window: int = 200,
        min_samples: int = 3,
        quantile: float = 0.5,
        exact_final: bool = True,
        human_quantile: float | None = None,
        end_aware: bool = False,
    ):
        self.min_samples = min_samples
        self.end_aware = end_aware
        self.quantile = quantile
        self.exact_final = exact_final
        # Waits on the user are heavy-tailed and may be the session's end;
        # predicting them with a high quantile keeps them out of CPU.
        self.human_quantile = human_quantile
        self.history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def observe(self, tool_name: str | None, duration_s: float) -> None:
        if tool_name is not None:
            self.history[tool_name].append(duration_s)

    def __call__(self, session: Session, k: int, gaps: list[float]) -> ReuseHint:
        tool = session.turns[k].tool_name
        if k == len(session.turns) - 1:
            if self.exact_final:
                return ReuseHint(session.session_id, k, None, final=True)
            tool = "human"
            if self.end_aware:
                tool = session.meta.get("end_next_tool") or "human"
        h = self.history.get(tool)
        q = self.quantile
        if tool == "human" and self.human_quantile is not None:
            q = self.human_quantile
        est = None
        if h and len(h) >= self.min_samples:
            if q == 0.5:
                est = statistics.median(h)
            else:
                xs = sorted(h)
                est = xs[min(len(xs) - 1, int(q * len(xs)))]
        return ReuseHint(session.session_id, k, est, tool=tool)


class ToolSessionEWMA(ToolMedian):
    """Post-response: EWMA of this session's earlier durations of the chosen
    tool, falling back to the global median of that tool. The same command
    tends to recur within a session (e.g. a test suite), which the global
    median of a broad tool like Bash cannot see."""

    def __init__(
        self,
        alpha: float = 0.5,
        window: int = 200,
        min_samples: int = 3,
        exact_final: bool = True,
    ):
        super().__init__(window, min_samples, exact_final=exact_final)
        self.alpha = alpha

    def __call__(self, session: Session, k: int, gaps: list[float]) -> ReuseHint:
        hint = super().__call__(session, k, gaps)
        if hint.final:
            return hint
        tool = session.turns[k].tool_name
        est = None
        for prev in session.turns[:k]:
            if prev.tool_name == tool:
                d = prev.tool_duration_s
                est = d if est is None else self.alpha * d + (1 - self.alpha) * est
        if est is None:
            return hint
        return ReuseHint(session.session_id, k, est, tool=hint.tool)


class ToolResidual(ToolMedian):
    """ToolMedian plus a residual table: for each elapsed time on the grid,
    the median remaining wait of past calls of that tool that ran at least
    that long. Waits are bimodal (instant calls vs. real work), so a call
    still running after ~1 s is likely in the slow mode; the table lets the
    cache see that instead of aging every late session the same way."""

    GRID = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)

    def __init__(self, window: int = 1000, min_survivors: int = 5, **kw):
        super().__init__(window, **kw)
        self.min_survivors = min_survivors

    def _table(self, tool: str, q: float) -> tuple[tuple[float, float], ...] | None:
        xs = sorted(self.history.get(tool, ()))
        table = []
        for e in self.GRID:
            rest = [x - e for x in xs if x > e]
            if len(rest) < self.min_survivors:
                break
            table.append((e, rest[min(len(rest) - 1, int(q * len(rest)))]))
        return tuple(table) or None

    def __call__(self, session: Session, k: int, gaps: list[float]) -> ReuseHint:
        hint = super().__call__(session, k, gaps)
        if hint.final:
            return hint
        tool = session.turns[k].tool_name
        q = self.quantile
        if k == len(session.turns) - 1 or tool == "human":
            tool = "human"
            if self.human_quantile is not None:
                q = self.human_quantile
        return replace(hint, residual=self._table(tool, q))


def is_post_response(predictor: Predictor) -> bool:
    return getattr(predictor, "post_response", False)


def get_predictor(name: str) -> Predictor:
    """Names: none, oracle, noisy:<sigma>, ewma[:<alpha>], tool[:<window>],
    toolq:<quantile> (a tool-duration quantile instead of the median),
    toolsess[:<alpha>] (session-local per-tool EWMA, global-median fallback),
    toolres[:<window>] (tool median plus elapsed-aware residual table).
    A `-nf` suffix on tool/toolq/toolsess drops the exact session-end flag;
    `tool-nfh` also predicts waits on the user with their 90th percentile.
    `tool-nfh2` is `tool-nfh` that, at a segment's artificial end (compaction
    or truncation), predicts the tool the trace called next instead of a user
    wait (see `ToolMedian.end_aware`). Both take `:<window>[:<human quantile>]`,
    e.g. `tool-nfh2:200:0.75`."""
    kind, _, arg = name.partition(":")
    human_q = None
    end_aware = False
    if kind.endswith("-nfh2"):
        kind, end_aware = kind.removesuffix("2"), True
    if kind.endswith("-nfh"):
        human_q, kind = 0.9, kind.removesuffix("h")
        arg, _, hq = arg.partition(":")
        if hq:
            human_q = float(hq)
    exact_final = not kind.endswith("-nf")
    kind = kind.removesuffix("-nf")
    if kind == "none":
        return no_hints
    if kind == "oracle":
        return oracle
    if kind == "noisy":
        return make_noisy_oracle(float(arg or 1.0))
    if kind == "ewma":
        return make_session_ewma(float(arg or 0.5))
    if kind == "tool":
        return ToolMedian(
            int(arg or 200),
            exact_final=exact_final,
            human_quantile=human_q,
            end_aware=end_aware,
        )
    if kind == "toolres":
        return ToolResidual(
            int(arg or 1000), exact_final=exact_final, human_quantile=human_q
        )
    if kind == "toolsess":
        return ToolSessionEWMA(float(arg or 0.5), exact_final=exact_final)
    if kind == "toolq":
        return ToolMedian(quantile=float(arg), exact_final=exact_final)
    raise ValueError(f"unknown predictor {name!r}")
