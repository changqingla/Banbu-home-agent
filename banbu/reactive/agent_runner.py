from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from openai import AsyncOpenAI

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings
from banbu.control.plane import ControlPlane
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import Turn

from .prompts import initial_messages, tool_followup_messages
from .protocol import (
    ReactiveAgentDecision,
    ReactiveAgentResult,
    ReactiveIntent,
    ReactiveToolCall,
    ReactiveToolRequest,
    parse_agent_decision,
    result_payload,
)
from .tools import ReactiveToolContext, ReactiveToolRegistry, ReactiveToolRunResult
from .tools.base import audit_tool_calls

log = logging.getLogger(__name__)


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
        tools: ReactiveToolRegistry | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._control = control
        self._audit = audit
        self._cache = cache
        self._scenes = scenes
        self._tools = tools or ReactiveToolRegistry.default()
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

        self._audit_turn(turn)

        messages = initial_messages(
            turn=turn,
            resolver=self._resolver,
            cache=self._cache,
            scenes=self._scenes,
            tool_specs=self._tools.prompt_specs(),
        )
        decision = await self._ask_agent(messages, turn=turn, phase="plan")
        self._audit_decision(turn, decision)

        if not decision.tool_calls:
            return self._complete(
                turn,
                intent=decision.intent,
                final_message=decision.final_message or self._default_message(decision.intent),
            )

        tool_result = await self._run_tool(turn, decision.intent, decision.tool_calls[0])
        if tool_result.final:
            return self._complete(
                turn,
                intent=tool_result.intent or decision.intent,
                final_message=tool_result.final_message,
                ok=tool_result.ok,
                scene_match=tool_result.scene_match,
                match=tool_result.match,
                device_mention=tool_result.device_mention,
                snapshot=tool_result.snapshot,
                execution=tool_result.execution,
                tool_calls=tool_result.tool_calls,
                error=tool_result.error,
                error_kind=tool_result.error_kind,
            )

        followup = tool_followup_messages(
            messages=messages,
            decision=decision,
            tool_calls=tool_result.tool_calls,
        )
        final_decision = await self._ask_agent(followup, turn=turn, phase="final")
        return self._complete(
            turn,
            intent=final_decision.intent,
            final_message=final_decision.final_message or tool_result.final_message,
            device_mention=tool_result.device_mention,
            snapshot=tool_result.snapshot,
            tool_calls=tool_result.tool_calls,
            error=tool_result.error,
            error_kind=tool_result.error_kind,
        )

    async def _run_tool(
        self,
        turn: Turn,
        intent: ReactiveIntent,
        request: ReactiveToolRequest,
    ) -> ReactiveToolRunResult:
        tool = self._tools.get(request.name)
        if tool is None:
            tool_call = ReactiveToolCall(
                name=request.name,
                args=request.args,
                result={"ok": False, "error": f"unsupported tool: {request.name!r}"},
            )
            audit_tool_calls(self._audit, turn, (tool_call,))
            return ReactiveToolRunResult(
                final=True,
                final_message="我还不能使用这个工具处理请求。",
                intent="unsupported",
                tool_calls=(tool_call,),
                error=tool_call.result["error"],
                error_kind="unsupported_tool",
            )

        ctx = ReactiveToolContext(
            turn=turn,
            intent=intent,
            resolver=self._resolver,
            cache=self._cache,
            control=self._control,
            audit=self._audit,
            scenes=self._scenes,
        )
        return await tool.run(ctx, request.args)

    async def _ask_agent(
        self,
        messages: list[dict[str, Any]],
        *,
        turn: Turn,
        phase: str,
    ) -> ReactiveAgentDecision:
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
            return ReactiveAgentDecision(
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
            return parse_agent_decision(content)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("reactive agent: invalid LLM response: %s", e)
            return ReactiveAgentDecision(
                intent="unsupported",
                tool_calls=(),
                final_message="我没能正确理解这次请求，请换一种说法。",
                raw=content,
            )

    def _audit_turn(self, turn: Turn) -> None:
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

    def _audit_decision(self, turn: Turn, decision: ReactiveAgentDecision) -> None:
        self._audit.write(
            "reactive_intent",
            {
                "intent": decision.intent,
                "tool_calls": [
                    {"name": request.name, "args": request.args}
                    for request in decision.tool_calls
                ],
                "final_message": decision.final_message,
            },
            trigger_id=turn.turn_id,
        )

    def _complete(
        self,
        turn: Turn,
        *,
        intent: ReactiveIntent,
        final_message: str,
        ok: bool = True,
        scene_match=None,
        match=None,
        device_mention=None,
        execution=None,
        snapshot=None,
        tool_calls: tuple[ReactiveToolCall, ...] = (),
        error: str | None = None,
        error_kind: str | None = None,
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

    def _default_message(self, intent: ReactiveIntent) -> str:
        if intent == "greeting":
            return "你好，我是 Banbu。你可以让我开关设备、查询设备状态，或者问我“你能做什么”。"
        if intent == "help":
            return "我可以帮你处理家里的即时请求，例如“打开玄关灯”或“玄关灯开着吗？”。"
        if intent == "clarification_needed":
            return "我需要你再说具体一点。"
        return "我还没理解这句话对应的家庭请求。"


__all__ = ["ReactiveAgentRunner", "ReactiveAgentResult", "result_payload"]
