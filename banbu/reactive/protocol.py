from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from banbu.control.plane import ExecuteResult
from banbu.state.snapshot_cache import Snapshot
from banbu.turn.model import Turn

from .matcher import ReactiveDeviceMention, ReactiveMatch
from .scene_matcher import SceneMatchCandidate

ReactiveIntent = Literal[
    "greeting",
    "help",
    "status_query",
    "control_request",
    "clarification_needed",
    "unsupported",
]

VALID_INTENTS: set[str] = {
    "greeting",
    "help",
    "status_query",
    "control_request",
    "clarification_needed",
    "unsupported",
}


@dataclass(frozen=True)
class ReactiveToolRequest:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ReactiveToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class ReactiveAgentDecision:
    intent: ReactiveIntent
    tool_calls: tuple[ReactiveToolRequest, ...]
    final_message: str
    raw: str


@dataclass(frozen=True)
class ReactiveAgentResult:
    ok: bool
    turn: Turn
    intent: ReactiveIntent
    final_message: str
    scene_match: SceneMatchCandidate | None = None
    match: ReactiveMatch | None = None
    device_mention: ReactiveDeviceMention | None = None
    snapshot: Snapshot | None = None
    execution: ExecuteResult | None = None
    tool_calls: tuple[ReactiveToolCall, ...] = ()
    error: str | None = None
    error_kind: str | None = None


def parse_agent_decision(raw: str) -> ReactiveAgentDecision:
    content = raw.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("reactive agent response must be a JSON object")

    intent = data.get("intent")
    if intent not in VALID_INTENTS:
        raise ValueError(f"invalid reactive intent: {intent!r}")

    tool_calls = data.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be an array")

    requests: list[ReactiveToolRequest] = []
    for call in tool_calls[:1]:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("args")
        if isinstance(name, str):
            requests.append(
                ReactiveToolRequest(
                    name=name,
                    args=args if isinstance(args, dict) else {},
                )
            )

    final_message = data.get("final_message")
    if not isinstance(final_message, str):
        final_message = ""

    return ReactiveAgentDecision(
        intent=intent,
        tool_calls=tuple(requests),
        final_message=final_message.strip(),
        raw=content,
    )


def result_payload(result: ReactiveAgentResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": result.ok,
        "turn_id": result.turn.turn_id,
        "conversation_id": result.turn.conversation_id,
        "intent": result.intent,
        "final_message": result.final_message,
    }
    if result.scene_match is not None:
        payload["scene_match"] = {
            "scene_id": result.scene_match.scene.scene_id,
            "score": result.scene_match.score,
            "reasons": list(result.scene_match.reasons),
        }
    if result.match is not None:
        payload["match"] = {
            "local_id": result.match.local_id,
            "friendly_name": result.match.device.spec.friendly_name,
            "action": result.match.action,
            "device_reasons": list(result.match.device_reasons),
        }
    if result.device_mention is not None:
        payload["device_mention"] = {
            "local_id": result.device_mention.local_id,
            "friendly_name": result.device_mention.device.spec.friendly_name,
            "device_reasons": list(result.device_mention.device_reasons),
        }
    if result.snapshot is not None:
        payload["snapshot"] = {
            "local_id": result.snapshot.local_id,
            "friendly_name": result.snapshot.friendly_name,
            "payload": result.snapshot.payload,
            "updated_at": result.snapshot.updated_at,
            "source": result.snapshot.source,
        }
    if result.execution is not None:
        payload["execution"] = {
            "ok": result.execution.ok,
            "local_id": result.execution.local_id,
            "action": result.execution.action,
            "payload": result.execution.payload,
            "error": result.execution.error,
            "deduped": result.execution.deduped,
        }
    if result.tool_calls:
        payload["tool_calls"] = [
            {"name": call.name, "args": call.args, "result": call.result}
            for call in result.tool_calls
        ]
    if result.error is not None:
        payload["error"] = result.error
        payload["error_kind"] = result.error_kind
    return payload
