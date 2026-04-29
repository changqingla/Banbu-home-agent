from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from banbu.agent.loop import AgentLoop
from banbu.agent.tools import TOOLS
from banbu.audit.log import AuditLog
from banbu.config.settings import Settings


class FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("fake LLM has no response left")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(*, content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    finish_reason = "tool_calls" if tool_calls else "stop"
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def _loop(tmp_path: Path, client: FakeClient) -> AgentLoop:
    settings = Settings(llm_model="test-model")
    return AgentLoop(settings, AuditLog(tmp_path / "audit.sqlite"), client=client)


@pytest.mark.asyncio
async def test_agent_executes_structured_execute_plan_tool_call(tmp_path: Path) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_execute",
                        "execute_plan",
                        {"local_id": 12, "action": "turn_on"},
                    )
                ]
            ),
            _response(content="done"),
        ]
    )
    loop = _loop(tmp_path, client)
    executed_calls: list[tuple[int, str, dict[str, Any] | None]] = []

    async def on_execute(local_id: int, action: str, params: dict[str, Any] | None) -> dict[str, Any]:
        executed_calls.append((local_id, action, params))
        return {"ok": True, "local_id": local_id, "action": action, "payload": {"state": "ON"}}

    async def on_read(local_id: int, fields: list[str] | None) -> dict[str, Any]:
        raise AssertionError(f"read_device_state was not expected: {local_id}, {fields}")

    result = await loop.run(
        [{"role": "system", "content": "policy"}],
        on_execute=on_execute,
        on_read=on_read,
        trigger_id="trg_tool_execute",
        scene_id="entry_auto_light_v1",
    )

    assert result.error is None
    assert result.iterations == 2
    assert result.executed == [{"ok": True, "local_id": 12, "action": "turn_on", "payload": {"state": "ON"}}]
    assert executed_calls == [(12, "turn_on", None)]
    assert client.completions.requests[0]["tools"] == TOOLS
    assert client.completions.requests[0]["tool_choice"] == "auto"
    tool_result = client.completions.requests[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_execute"
    assert json.loads(tool_result["content"])["ok"] is True


@pytest.mark.asyncio
async def test_agent_handles_read_device_state_before_execute_plan(tmp_path: Path) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_read",
                        "read_device_state",
                        {"local_id": 12, "fields": ["state"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "call_execute",
                        "execute_plan",
                        {"local_id": 12, "action": "turn_on", "params": {"brightness": 200}},
                    )
                ]
            ),
            _response(content="done"),
        ]
    )
    loop = _loop(tmp_path, client)
    read_calls: list[tuple[int, list[str] | None]] = []
    executed_calls: list[tuple[int, str, dict[str, Any] | None]] = []

    async def on_read(local_id: int, fields: list[str] | None) -> dict[str, Any]:
        read_calls.append((local_id, fields))
        return {"ok": True, "local_id": local_id, "payload": {"state": "OFF"}}

    async def on_execute(local_id: int, action: str, params: dict[str, Any] | None) -> dict[str, Any]:
        executed_calls.append((local_id, action, params))
        return {"ok": True, "local_id": local_id, "action": action, "payload": {"state": "ON"}}

    result = await loop.run(
        [{"role": "system", "content": "policy"}],
        on_execute=on_execute,
        on_read=on_read,
        trigger_id="trg_read_then_execute",
        scene_id="entry_auto_light_v1",
    )

    assert result.error is None
    assert result.iterations == 3
    assert read_calls == [(12, ["state"])]
    assert executed_calls == [(12, "turn_on", {"brightness": 200})]
    first_tool_result = client.completions.requests[1]["messages"][-1]
    assert json.loads(first_tool_result["content"]) == {
        "local_id": 12,
        "ok": True,
        "payload": {"state": "OFF"},
    }


@pytest.mark.asyncio
async def test_agent_does_not_execute_text_json_without_tool_call(tmp_path: Path) -> None:
    client = FakeClient([_response(content='[{"local_id": 12, "action": "turn_on"}]')])
    loop = _loop(tmp_path, client)

    async def on_execute(local_id: int, action: str, params: dict[str, Any] | None) -> dict[str, Any]:
        raise AssertionError(f"text JSON must not execute: {local_id}, {action}, {params}")

    async def on_read(local_id: int, fields: list[str] | None) -> dict[str, Any]:
        raise AssertionError(f"read_device_state was not expected: {local_id}, {fields}")

    result = await loop.run(
        [{"role": "system", "content": "policy"}],
        on_execute=on_execute,
        on_read=on_read,
        trigger_id="trg_no_text_fallback",
        scene_id="entry_auto_light_v1",
    )

    assert result.error is None
    assert result.executed == []
    assert result.final_message == '[{"local_id": 12, "action": "turn_on"}]'
