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
    )


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
