"""Reuse hints attached to requests.

A hint says: "after this request finishes, the same session will need this
prefix again in about `expected_reuse_s` seconds" (or never, if `final`).

Two transports carry the same payload:
  * HTTP: `kv_transfer_params["kvplace"] = {...}` — supported by the OpenAI
    server today, so this is what the replay client uses.
  * Python engine API: a `KvHintsEnvelope` action with
    `action_type == ACTION_TYPE` (vllm/v1/kv_hints/protocol.py). Not yet
    exposed over HTTP.

This module must stay importable without vLLM (the simulator uses it).
"""

import bisect
from dataclasses import asdict, dataclass
from typing import Any

HINT_KEY = "kvplace"
ACTION_TYPE = "kvplace.reuse_horizon"
ACTION_VERSION = "1"


@dataclass(frozen=True)
class ReuseHint:
    session_id: str
    turn: int
    # Seconds from this request's completion to the session's next request.
    # None means "no prediction".
    expected_reuse_s: float | None = None
    # True if the session ends after this request (prefix will not be reused).
    final: bool = False
    # Optional (elapsed_s, expected_remaining_s) breakpoints: the expected
    # remaining wait once the session has already waited elapsed_s. Lets the
    # cache re-rank sessions as they wait instead of trusting one estimate.
    residual: tuple[tuple[float, float], ...] | None = None
    # Tool the agent is running until its next request ("human" = waiting on
    # the user). None means unknown. Optional; older payloads omit it.
    tool: str | None = None
    # Optional evenly spaced quantiles of this tool's recent wait times, so the
    # cache can estimate the chance the session returns within a horizon.
    quantiles: tuple[float, ...] | None = None

    def to_kv_transfer_params(self) -> dict[str, Any]:
        return {HINT_KEY: asdict(self)}

    def to_kv_hint_action(self) -> dict[str, Any]:
        """Fields for vllm.v1.kv_hints.KvHintAction."""
        return {
            "action_id": f"{self.session_id}:{self.turn}",
            "action_type": ACTION_TYPE,
            "action_version": ACTION_VERSION,
            "payload": asdict(self),
        }


def _from_payload(payload: Any) -> ReuseHint | None:
    if not isinstance(payload, dict) or "session_id" not in payload:
        return None
    reuse = payload.get("expected_reuse_s")
    return ReuseHint(
        session_id=str(payload["session_id"]),
        turn=int(payload.get("turn", 0)),
        expected_reuse_s=None if reuse is None else float(reuse),
        final=bool(payload.get("final", False)),
        residual=_residual(payload.get("residual")),
        tool=None if payload.get("tool") is None else str(payload["tool"]),
        quantiles=(
            tuple(float(x) for x in payload["quantiles"])
            if payload.get("quantiles")
            else None
        ),
    )


def _residual(raw: Any) -> tuple[tuple[float, float], ...] | None:
    if not raw:
        return None
    return tuple((float(e), float(r)) for e, r in raw)


def return_probability(
    quantiles: tuple[float, ...], elapsed_s: float, horizon_s: float
) -> float:
    """P(wait <= elapsed + horizon | wait > elapsed), from evenly spaced
    quantiles of the wait distribution. 0 once the wait outlasts them all."""

    def cdf(x: float) -> float:
        return bisect.bisect_right(quantiles, x) / len(quantiles)

    alive = 1.0 - cdf(elapsed_s)
    if alive <= 0.0:
        return 0.0
    return (cdf(elapsed_s + horizon_s) - cdf(elapsed_s)) / alive


def expected_remaining(
    residual: tuple[tuple[float, float], ...], elapsed_s: float
) -> float:
    """Expected remaining wait after `elapsed_s`, interpolated between
    breakpoints; past the last one, assume it lasts as long again."""
    prev_e, prev_r = residual[0]
    if elapsed_s <= prev_e:
        return prev_r
    for e, r in residual[1:]:
        if elapsed_s <= e:
            f = (elapsed_s - prev_e) / (e - prev_e)
            return prev_r + f * (r - prev_r)
        prev_e, prev_r = e, r
    return max(prev_r, elapsed_s)


def parse_hint(req_context: Any) -> ReuseHint | None:
    """Extract a ReuseHint from a vLLM ReqContext, preferring kv_hints."""
    envelope = getattr(req_context, "kv_hints", None)
    if envelope is not None:
        for action in envelope.actions:
            if action.action_type == ACTION_TYPE:
                return _from_payload(action.payload)
    params = getattr(req_context, "kv_transfer_params", None)
    if params:
        return _from_payload(params.get(HINT_KEY))
    return None
