from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings
from banbu.control.plane import ControlPlane, ExecuteResult
from banbu.devices.definition import ResolvedDevice, effective_actions
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import Snapshot, SnapshotCache
from banbu.turn.model import Turn

from .matcher import (
    ReactiveDeviceMention,
    ReactiveMatch,
    ReactiveMatchError,
    match_device_action,
    match_device_mention,
)
from .scene_matcher import SceneMatchCandidate, SceneMatchError, select_scene_match

log = logging.getLogger(__name__)

ReactiveIntent = Literal[
    "greeting",
    "help",
    "status_query",
    "control_request",
    "clarification_needed",
    "unsupported",
]

_VALID_INTENTS = {
    "greeting",
    "help",
    "status_query",
    "control_request",
    "clarification_needed",
    "unsupported",
}

_SYSTEM_PROMPT = """\
You are Banbu, a bounded smart-home reactive agent for IM user turns.

Return exactly one JSON object. Do not use markdown.

Schema:
{
  "intent": "greeting|help|status_query|control_request|clarification_needed|unsupported",
  "tool_calls": [
    {"name": "get_device_state", "args": {"local_id": 2}},
    {"name": "list_relevant_devices", "args": {}},
    {"name": "execute_plan", "args": {"local_id": 2, "action": "turn_on", "params": null}}
  ],
  "final_message": "Short Chinese reply if no tool is needed or after using provided tool results."
}

Rules:
- You may call at most one tool per response.
- Use get_device_state/list_relevant_devices for status questions.
- Use execute_plan only when the user explicitly asks for a concrete device action.
- If the device/action is ambiguous, ask a clarification question and do not call execute_plan.
- Do not invent local_id, action, device names, or state values. Use only the device context.
- All device writes are executed by backend policy; you only propose execute_plan.
- Reply in concise Chinese.
"""


@dataclass(frozen=True)
class ReactiveToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


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


@dataclass(frozen=True)
class _AgentDecision:
    intent: ReactiveIntent
    tool_calls: tuple[dict[str, Any], ...]
    final_message: str
    raw: str


def _display_name(device: ResolvedDevice) -> str:
    return device.spec.aliases[0] if device.spec.aliases else device.spec.friendly_name


def _state_label(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.upper()
        if normalized == "ON":
            return "开着"
        if normalized == "OFF":
            return "关着"
    if isinstance(value, bool):
        return "开着" if value else "关着"
    return repr(value)


def _snapshot_summary(device: ResolvedDevice, snapshot: Snapshot | None) -> str:
    name = _display_name(device)
    if snapshot is None:
        return f"我现在还没有{name}的状态快照。"

    payload = snapshot.payload
    if "state" in payload:
        return f"{name}现在是{_state_label(payload['state'])}。"

    care_fields = [field for field in device.spec.care_fields if field in payload]
    if care_fields:
        facts = "，".join(f"{field}={payload[field]!r}" for field in care_fields[:4])
        return f"{name}当前状态：{facts}。"

    if payload:
        facts = "，".join(f"{key}={value!r}" for key, value in list(sorted(payload.items()))[:4])
        return f"{name}当前状态：{facts}。"

    return f"我有{name}的设备记录，但当前快照是空的。"


def _device_context(resolver: DeviceResolver, cache: SnapshotCache) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for device in resolver.all():
        snapshot = cache.get(device.local_id)
        devices.append(
            {
                "local_id": device.local_id,
                "friendly_name": device.spec.friendly_name,
                "display_name": _display_name(device),
                "room": device.spec.room,
                "role": device.spec.role,
                "aliases": list(device.spec.aliases),
                "actions": sorted(effective_actions(device.spec)),
                "care_fields": list(device.spec.care_fields),
                "snapshot": snapshot.payload if snapshot is not None else None,
            }
        )
    return devices


def _parse_decision(raw: str) -> _AgentDecision:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("reactive agent response must be a JSON object")

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        raise ValueError(f"invalid reactive intent: {intent!r}")

    tool_calls = data.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be an array")
    normalized_calls: list[dict[str, Any]] = []
    for call in tool_calls[:1]:
        if isinstance(call, dict):
            normalized_calls.append(call)

    final_message = data.get("final_message")
    if not isinstance(final_message, str):
        final_message = ""

    return _AgentDecision(
        intent=intent,
        tool_calls=tuple(normalized_calls),
        final_message=final_message.strip(),
        raw=raw,
    )


class ReactiveAgentRunner:
    """LLM-driven, bounded reactive user-turn loop for IM interactions."""

    def __init__(
        self,
        *,
        settings: Settings,
        resolver: DeviceResolver,
        control: ControlPlane,
        audit: AuditLog,
        cache: SnapshotCache,
        scenes: list[Scene] | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._control = control
        self._audit = audit
        self._cache = cache
        self._scenes = scenes
        self._client = client or AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "EMPTY",
            timeout=settings.llm_timeout_seconds,
            http_client=httpx.AsyncClient(
                timeout=settings.llm_timeout_seconds,
                trust_env=False,
            ),
        )

    async def run(self, turn: Turn) -> ReactiveAgentResult:
        if turn.thread_type != "reactive":
            raise ValueError(f"ReactiveAgentRunner requires reactive turns, got {turn.thread_type!r}")
        if turn.utterance is None:
            raise ValueError("reactive turn must include utterance")

        self._audit.write(
            "reactive_turn",
            {
                "turn_id": turn.turn_id,
                "conversation_id": turn.conversation_id,
                "home_id": turn.home_id,
                "user_id": turn.user_id,
                "source": turn.source,
                "utterance": turn.utterance,
                "runner": "reactive_agent_llm",
            },
            trigger_id=turn.turn_id,
        )

        messages = self._initial_messages(turn)
        decision = await self._ask_agent(messages, turn=turn, phase="plan")
        self._audit.write(
            "reactive_intent",
            {
                "intent": decision.intent,
                "tool_calls": list(decision.tool_calls),
                "final_message": decision.final_message,
            },
            trigger_id=turn.turn_id,
        )

        if not decision.tool_calls:
            return self._complete(
                turn,
                intent=decision.intent,
                final_message=decision.final_message or self._default_message(decision.intent),
            )

        tool_call = decision.tool_calls[0]
        result = await self._run_tool(turn, decision.intent, tool_call)
        if result.execution is not None or tool_call.get("name") == "execute_plan":
            return result

        followup = self._tool_followup_messages(messages, decision, result.tool_calls)
        final_decision = await self._ask_agent(followup, turn=turn, phase="final")
        return self._complete(
            turn,
            intent=final_decision.intent,
            final_message=final_decision.final_message or result.final_message,
            device_mention=result.device_mention,
            snapshot=result.snapshot,
            tool_calls=result.tool_calls,
            error=result.error,
            error_kind=result.error_kind,
        )

    def _initial_messages(self, turn: Turn) -> list[dict[str, Any]]:
        user_payload = {
            "home_id": turn.home_id,
            "user_id": turn.user_id,
            "source": turn.source,
            "utterance": turn.utterance,
            "devices": _device_context(self._resolver, self._cache),
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "name": scene.name,
                    "intent": scene.intent,
                    "referenced_devices": list(scene.all_referenced_devices()),
                }
                for scene in self._scenes or []
            ],
        }
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
        ]

    def _tool_followup_messages(
        self,
        messages: list[dict[str, Any]],
        decision: _AgentDecision,
        tool_calls: tuple[ReactiveToolCall, ...],
    ) -> list[dict[str, Any]]:
        tool_payload = {
            "previous_decision": {
                "intent": decision.intent,
                "tool_calls": list(decision.tool_calls),
                "final_message": decision.final_message,
            },
            "tool_results": [
                {"name": call.name, "args": call.args, "result": call.result}
                for call in tool_calls
            ],
            "instruction": "Use the tool result to produce the final user-facing Chinese reply. Do not call another tool.",
        }
        return [
            *messages,
            {"role": "assistant", "content": decision.raw},
            {"role": "user", "content": json.dumps(tool_payload, ensure_ascii=False, sort_keys=True)},
        ]

    async def _ask_agent(self, messages: list[dict[str, Any]], *, turn: Turn, phase: str) -> _AgentDecision:
        self._audit.write(
            "reactive_agent_request",
            {"phase": phase, "messages": messages, "model": self._settings.llm_model},
            trigger_id=turn.turn_id,
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except Exception as e:
            log.exception("reactive agent: LLM request failed")
            self._audit.write(
                "reactive_agent_response",
                {"phase": phase, "error": str(e)},
                trigger_id=turn.turn_id,
            )
            return _AgentDecision(
                intent="unsupported",
                tool_calls=(),
                final_message="我现在没法完成这次请求，稍后再试一下。",
                raw="",
            )

        content = response.choices[0].message.content or ""
        self._audit.write(
            "reactive_agent_response",
            {
                "phase": phase,
                "finish_reason": response.choices[0].finish_reason,
                "content": content,
            },
            trigger_id=turn.turn_id,
        )
        try:
            return _parse_decision(content)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("reactive agent: invalid LLM response: %s", e)
            return _AgentDecision(
                intent="unsupported",
                tool_calls=(),
                final_message="我没能正确理解这次请求，请换一种说法。",
                raw=content,
            )

    async def _run_tool(self, turn: Turn, intent: ReactiveIntent, call: dict[str, Any]) -> ReactiveAgentResult:
        name = call.get("name")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}

        if name == "get_device_state":
            return self._tool_get_device_state(turn, intent, args)
        if name == "list_relevant_devices":
            return self._tool_list_relevant_devices(turn, intent, args)
        if name == "execute_plan":
            return await self._tool_execute_plan(turn, intent, args)

        tool_call = ReactiveToolCall(
            name=str(name),
            args=args,
            result={"ok": False, "error": f"unsupported tool: {name!r}"},
        )
        self._audit_tool_calls(turn, [tool_call], None)
        return self._complete(
            turn,
            intent="unsupported",
            final_message="我还不能使用这个工具处理请求。",
            tool_calls=(tool_call,),
            error=tool_call.result["error"],
            error_kind="unsupported_tool",
        )

    def _tool_get_device_state(
        self,
        turn: Turn,
        intent: ReactiveIntent,
        args: dict[str, Any],
    ) -> ReactiveAgentResult:
        mention = self._safe_device_mention(turn, args)
        snapshot = self._cache.get(mention.local_id) if mention is not None else None
        result = {
            "ok": mention is not None,
            "local_id": mention.local_id if mention is not None else args.get("local_id"),
            "friendly_name": mention.device.spec.friendly_name if mention is not None else None,
            "display_name": _display_name(mention.device) if mention is not None else None,
            "payload": snapshot.payload if snapshot is not None else None,
            "error": None if mention is not None else "could not safely identify the requested device",
        }
        tool_call = ReactiveToolCall(name="get_device_state", args=args, result=result)
        self._audit_tool_calls(turn, [tool_call], None)
        return self._complete(
            turn,
            intent=intent,
            final_message=_snapshot_summary(mention.device, snapshot) if mention is not None else "我不确定你要查哪个设备。",
            device_mention=mention,
            snapshot=snapshot,
            tool_calls=(tool_call,),
            error=result["error"],
            error_kind=None if mention is not None else "unknown_device",
            audit_final=False,
        )

    def _tool_list_relevant_devices(
        self,
        turn: Turn,
        intent: ReactiveIntent,
        args: dict[str, Any],
    ) -> ReactiveAgentResult:
        snapshots = self._cache.all()
        result = {
            "ok": True,
            "count": len(snapshots),
            "devices": [
                {
                    "local_id": snap.local_id,
                    "friendly_name": snap.friendly_name,
                    "payload": snap.payload,
                }
                for snap in snapshots
            ],
        }
        tool_call = ReactiveToolCall(name="list_relevant_devices", args=args, result=result)
        self._audit_tool_calls(turn, [tool_call], None)
        if not snapshots:
            message = "我现在还没有家里设备的状态快照。"
        else:
            summaries = []
            for snap in snapshots[:5]:
                device = self._resolver.by_local_id(snap.local_id)
                name = _display_name(device) if device is not None else snap.friendly_name
                if "state" in snap.payload:
                    summaries.append(f"{name}{_state_label(snap.payload['state'])}")
                else:
                    summaries.append(name)
            suffix = "；还有更多设备。" if len(snapshots) > 5 else "。"
            message = "现在我能看到：" + "，".join(summaries) + suffix
        return self._complete(turn, intent=intent, final_message=message, tool_calls=(tool_call,))

    async def _tool_execute_plan(
        self,
        turn: Turn,
        intent: ReactiveIntent,
        args: dict[str, Any],
    ) -> ReactiveAgentResult:
        safe_match, guard_error, guard_kind = self._safe_control_match(turn, args)
        if safe_match is None:
            tool_call = ReactiveToolCall(
                name="execute_plan",
                args=args,
                result={"ok": False, "error": guard_error, "guard": guard_kind},
            )
            self._audit_tool_calls(turn, [tool_call], None)
            return self._complete(
                turn,
                intent="clarification_needed",
                final_message=self._clarification_message(guard_kind, guard_error),
                tool_calls=(tool_call,),
                error=guard_error,
                error_kind=guard_kind,
            )

        scene_match = self._select_scene_for_control(turn, safe_match)
        execution = await self._control.execute(
            safe_match.local_id,
            safe_match.action,
            args.get("params"),
            trigger_id=turn.turn_id,
            scene_id=scene_match.scene.scene_id if scene_match is not None else None,
            actor="reactive",
            home_id=turn.home_id,
            user_id=turn.user_id,
        )
        tool_call = ReactiveToolCall(
            name="execute_plan",
            args={"local_id": safe_match.local_id, "action": safe_match.action, "params": args.get("params")},
            result={
                "ok": execution.ok,
                "local_id": execution.local_id,
                "action": execution.action,
                "payload": execution.payload,
                "error": execution.error,
                "deduped": execution.deduped,
            },
        )
        self._audit_tool_calls(turn, [tool_call], scene_match)

        if execution.ok:
            action_label = "打开" if safe_match.action == "turn_on" else "关闭" if safe_match.action == "turn_off" else safe_match.action
            name = _display_name(safe_match.device)
            if execution.deduped:
                final_message = f"已收到，{name} 的{action_label}指令刚刚执行过。"
            else:
                final_message = f"已{action_label}{name}。"
            error = None
            error_kind = None
        else:
            final_message = f"没能完成这次请求：{execution.error or '执行失败'}"
            error = execution.error
            error_kind = "execute_failed"

        return self._complete(
            turn,
            intent=intent,
            ok=execution.ok,
            final_message=final_message,
            scene_match=scene_match,
            match=safe_match,
            execution=execution,
            tool_calls=(tool_call,),
            error=error,
            error_kind=error_kind,
        )

    def _safe_control_match(self, turn: Turn, args: dict[str, Any]) -> tuple[ReactiveMatch | None, str | None, str | None]:
        try:
            match = match_device_action(turn.utterance or "", self._resolver)
        except ReactiveMatchError as e:
            return None, str(e), e.kind

        requested_local_id = args.get("local_id")
        requested_action = args.get("action")
        if requested_local_id is not None and int(requested_local_id) != match.local_id:
            return None, "LLM selected a device that was not explicitly matched from the utterance", "unsafe_device"
        if requested_action is not None and str(requested_action) != match.action:
            return None, "LLM selected an action that was not explicitly matched from the utterance", "unsafe_action"
        return match, None, None

    def _safe_device_mention(self, turn: Turn, args: dict[str, Any]) -> ReactiveDeviceMention | None:
        try:
            mention = match_device_mention(turn.utterance or "", self._resolver)
        except ReactiveMatchError:
            local_id = args.get("local_id")
            if local_id is None:
                return None
            device = self._resolver.by_local_id(int(local_id))
            if device is None:
                return None
            return ReactiveDeviceMention(device=device, device_reasons=("llm_selected_local_id",))

        requested_local_id = args.get("local_id")
        if requested_local_id is not None and int(requested_local_id) != mention.local_id:
            return None
        return mention

    def _select_scene_for_control(self, turn: Turn, match: ReactiveMatch) -> SceneMatchCandidate | None:
        if self._scenes is None:
            return None
        try:
            scene_match = select_scene_match(turn.utterance or "", self._scenes, self._resolver)
        except SceneMatchError as e:
            self._audit.write(
                "reactive_scene_match",
                {
                    "ok": False,
                    "kind": e.kind,
                    "error": str(e),
                    "candidates": list(e.candidates),
                    "blocking": False,
                },
                trigger_id=turn.turn_id,
            )
            return None

        referenced = scene_match.scene.all_referenced_devices()
        if match.device.spec.friendly_name not in referenced:
            self._audit.write(
                "reactive_scene_match",
                {
                    "ok": False,
                    "kind": "device_outside_scene",
                    "scene_id": scene_match.scene.scene_id,
                    "local_id": match.local_id,
                    "friendly_name": match.device.spec.friendly_name,
                    "blocking": False,
                },
                trigger_id=turn.turn_id,
                scene_id=scene_match.scene.scene_id,
            )
            return None

        self._audit.write(
            "reactive_scene_match",
            {
                "ok": True,
                "scene_id": scene_match.scene.scene_id,
                "score": scene_match.score,
                "reasons": list(scene_match.reasons),
                "blocking": False,
            },
            trigger_id=turn.turn_id,
            scene_id=scene_match.scene.scene_id,
        )
        return scene_match

    def _clarification_message(self, error_kind: str | None, error: str | None) -> str:
        if error_kind == "ambiguous_device":
            return "我不确定你指的是哪个设备，请说得更具体一点。"
        if error_kind in {"unknown_device", "unsafe_device"}:
            return "我没找到你说的设备。你可以换成设备别名，比如“玄关灯”。"
        if error_kind in {"unknown_action", "unsafe_action"}:
            return "我还没判断出你要执行什么动作。你可以说“打开玄关灯”或“关闭玄关灯”。"
        if error_kind == "unsupported_action":
            return "这个设备目前不支持你说的动作。"
        return error or "我还没理解这句话对应的家庭请求。"

    def _default_message(self, intent: ReactiveIntent) -> str:
        if intent == "greeting":
            return "你好，我是 Banbu。你可以让我开关设备、查询设备状态，或者问我“你能做什么”。"
        if intent == "help":
            return "我可以帮你处理家里的即时请求，例如“打开玄关灯”或“玄关灯开着吗？”。"
        if intent == "clarification_needed":
            return "我需要你再说具体一点。"
        return "我还没理解这句话对应的家庭请求。"

    def _audit_tool_calls(
        self,
        turn: Turn,
        tool_calls: list[ReactiveToolCall],
        scene_match: SceneMatchCandidate | None,
    ) -> None:
        for call in tool_calls:
            self._audit.write(
                "reactive_tool_call",
                {"name": call.name, "args": call.args},
                trigger_id=turn.turn_id,
                scene_id=scene_match.scene.scene_id if scene_match is not None else None,
            )
            self._audit.write(
                "reactive_tool_result",
                {"name": call.name, "result": call.result},
                trigger_id=turn.turn_id,
                scene_id=scene_match.scene.scene_id if scene_match is not None else None,
            )

    def _complete(
        self,
        turn: Turn,
        *,
        intent: ReactiveIntent,
        final_message: str,
        ok: bool = True,
        scene_match: SceneMatchCandidate | None = None,
        match: ReactiveMatch | None = None,
        device_mention: ReactiveDeviceMention | None = None,
        execution: ExecuteResult | None = None,
        snapshot: Snapshot | None = None,
        tool_calls: tuple[ReactiveToolCall, ...] = (),
        error: str | None = None,
        error_kind: str | None = None,
        audit_final: bool = True,
    ) -> ReactiveAgentResult:
        result = ReactiveAgentResult(
            ok=ok,
            turn=turn,
            intent=intent,
            final_message=final_message,
            scene_match=scene_match,
            match=match,
            device_mention=device_mention,
            snapshot=snapshot,
            execution=execution,
            tool_calls=tool_calls,
            error=error,
            error_kind=error_kind,
        )
        if audit_final:
            self._audit.write(
                "reactive_final",
                {
                    "ok": result.ok,
                    "intent": result.intent,
                    "final_message": result.final_message,
                    "error": result.error,
                    "error_kind": result.error_kind,
                },
                trigger_id=turn.turn_id,
                scene_id=scene_match.scene.scene_id if scene_match is not None else None,
            )
        return result


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
