from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings

log = logging.getLogger(__name__)

ExecuteHandler = Callable[[int, str, dict[str, Any] | None], Awaitable[dict[str, Any]]]


@dataclass
class AgentResult:
    iterations: int
    executed: list[dict[str, Any]] = field(default_factory=list)
    final_message: str | None = None
    error: str | None = None


def _extract_actions(text: str) -> list[dict]:
    """Extract JSON array of actions from LLM output."""
    # Try ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Fall back: first [...] in text
    m = re.search(r"(\[.*?\])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    return []


class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        audit: AuditLog,
        *,
        client: AsyncOpenAI | None = None,
        max_iterations: int = 5,  # kept for API compatibility, unused
    ) -> None:
        self._settings = settings
        self._audit = audit
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
        on_read: Any = None,  # kept for API compatibility, unused
        trigger_id: str | None = None,
        scene_id: str | None = None,
    ) -> AgentResult:
        self._audit.write(
            "agent_request",
            {"messages": messages, "model": self._settings.llm_model},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )

        log.info("agent: LLM request\n%s", json.dumps(messages, ensure_ascii=False, indent=2))

        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except Exception as e:
            log.exception("agent: chat completion failed")
            self._audit.write("agent_response", {"error": str(e)}, trigger_id=trigger_id, scene_id=scene_id)
            return AgentResult(iterations=1, error=str(e))

        content = resp.choices[0].message.content or ""
        log.info("agent: LLM response: %r", content)
        self._audit.write(
            "agent_response",
            {"finish_reason": resp.choices[0].finish_reason, "content": content},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )

        try:
            actions = _extract_actions(content)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("agent: failed to parse actions from response: %s", e)
            return AgentResult(iterations=1, final_message=content, error=f"json_parse_error: {e}")

        executed: list[dict[str, Any]] = []
        for act in actions:
            local_id = act.get("local_id")
            action = act.get("action")
            if local_id is None or action is None:
                log.warning("agent: skipping malformed action: %s", act)
                continue
            self._audit.write(
                "tool_call", {"name": "execute_plan", "args": act},
                trigger_id=trigger_id, scene_id=scene_id,
            )
            try:
                result = await on_execute(int(local_id), str(action), act.get("params"))
                if result.get("ok"):
                    executed.append(result)
            except Exception as e:
                log.exception("agent: execute failed for action %s", act)
                result = {"ok": False, "error": str(e)}
            self._audit.write(
                "tool_result", {"name": "execute_plan", "result": result},
                trigger_id=trigger_id, scene_id=scene_id,
            )

        log.info("agent: done, executed=%d", len(executed))
        return AgentResult(iterations=1, executed=executed, final_message=content)
