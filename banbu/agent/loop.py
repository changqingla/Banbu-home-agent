from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings

from .tools import TOOLS

log = logging.getLogger(__name__)

ExecuteHandler = Callable[[int, str, dict[str, Any] | None], Awaitable[dict[str, Any]]]
ReadHandler = Callable[[int, list[str] | None], Awaitable[dict[str, Any]]]


@dataclass
class AgentResult:
    iterations: int
    executed: list[dict[str, Any]] = field(default_factory=list)
    final_message: str | None = None
    error: str | None = None


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"tool arguments must be a JSON object, got {type(data).__name__}")
    return data


def _assistant_message(message: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"role": "assistant", "content": message.content}
    tool_calls = message.tool_calls or []
    if tool_calls:
        item["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return item


def _tool_message(tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
    }


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("fields must be a list of strings")
    return value


class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        audit: AuditLog,
        *,
        client: Any | None = None,
        max_iterations: int = 5,
    ) -> None:
        self._settings = settings
        self._audit = audit
        self._max_iterations = max_iterations
        self._client = client or AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "EMPTY",
            timeout=settings.llm_timeout_seconds,
            http_client=httpx.AsyncClient(
                timeout=settings.llm_timeout_seconds,
                trust_env=False,
            ),
        )

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        on_execute: ExecuteHandler,
        on_read: ReadHandler,
        trigger_id: str | None = None,
        scene_id: str | None = None,
    ) -> AgentResult:
        conversation = [dict(m) for m in messages]
        executed: list[dict[str, Any]] = []

        for iteration in range(1, self._max_iterations + 1):
            request_messages = [dict(m) for m in conversation]
            self._audit.write(
                "agent_request",
                {
                    "iteration": iteration,
                    "messages": request_messages,
                    "model": self._settings.llm_model,
                    "tools": TOOLS,
                },
                trigger_id=trigger_id,
                scene_id=scene_id,
            )

            log.info(
                "agent: LLM request iteration=%d\n%s",
                iteration,
                json.dumps(request_messages, ensure_ascii=False, indent=2),
            )
            try:
                resp = await self._client.chat.completions.create(
                    model=self._settings.llm_model,
                    messages=request_messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=self._settings.llm_temperature,
                    max_tokens=self._settings.llm_max_tokens,
                )
            except Exception as e:
                log.exception("agent: chat completion failed")
                self._audit.write(
                    "agent_response",
                    {"iteration": iteration, "error": str(e)},
                    trigger_id=trigger_id,
                    scene_id=scene_id,
                )
                return AgentResult(iterations=iteration, executed=executed, error=str(e))

            choice = resp.choices[0]
            message = choice.message
            tool_calls = list(message.tool_calls or [])
            content = message.content or ""
            log.info(
                "agent: LLM response iteration=%d finish=%s content=%r tool_calls=%d",
                iteration,
                choice.finish_reason,
                content,
                len(tool_calls),
            )
            self._audit.write(
                "agent_response",
                {
                    "iteration": iteration,
                    "finish_reason": choice.finish_reason,
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                        for call in tool_calls
                    ],
                },
                trigger_id=trigger_id,
                scene_id=scene_id,
            )

            conversation.append(_assistant_message(message))
            if not tool_calls:
                log.info("agent: done iterations=%d executed=%d", iteration, len(executed))
                return AgentResult(iterations=iteration, executed=executed, final_message=content)

            for call in tool_calls:
                result = await self._dispatch_tool_call(
                    call.function.name,
                    call.function.arguments,
                    on_execute=on_execute,
                    on_read=on_read,
                    trigger_id=trigger_id,
                    scene_id=scene_id,
                )
                if call.function.name == "execute_plan" and result.get("ok"):
                    executed.append(result)
                conversation.append(_tool_message(call.id, result))

        error = f"max_iterations_exceeded: {self._max_iterations}"
        log.warning("agent: %s", error)
        return AgentResult(iterations=self._max_iterations, executed=executed, error=error)

    async def _dispatch_tool_call(
        self,
        name: str,
        raw_args: str | None,
        *,
        on_execute: ExecuteHandler,
        on_read: ReadHandler,
        trigger_id: str | None,
        scene_id: str | None,
    ) -> dict[str, Any]:
        try:
            args = _parse_tool_args(raw_args)
            self._audit.write(
                "tool_call",
                {"name": name, "args": args},
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            if name == "execute_plan":
                result = await on_execute(
                    int(args["local_id"]),
                    str(args["action"]),
                    args.get("params"),
                )
            elif name == "read_device_state":
                result = await on_read(int(args["local_id"]), _string_list(args.get("fields")))
            else:
                result = {"ok": False, "error": f"unknown tool: {name}"}
        except Exception as e:
            log.exception("agent: tool call failed name=%s args=%r", name, raw_args)
            result = {"ok": False, "error": str(e)}

        self._audit.write(
            "tool_result",
            {"name": name, "result": result},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )
        return result
