from types import SimpleNamespace

import pytest

from banbu.agent.loop import AgentLoop
from banbu.audit.log import AuditLog
from banbu.config.settings import Settings


class FakeCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='```json\n[{"local_id": 2, "action": "turn_on"}]\n```'
                    ),
                )
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


@pytest.mark.asyncio
async def test_agent_loop_extracts_json_action_array_and_executes(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.sqlite")
    loop = AgentLoop(Settings(llm_model="test-model"), audit, client=FakeClient())
    calls: list[tuple[int, str, dict | None]] = []

    async def on_execute(local_id: int, action: str, params: dict | None) -> dict:
        calls.append((local_id, action, params))
        return {"ok": True, "local_id": local_id, "action": action, "payload": {"state": "ON"}}

    result = await loop.run(
        [{"role": "user", "content": "turn on"}],
        on_execute=on_execute,
        trigger_id="trg_1",
        scene_id="scene_a",
    )

    assert result.error is None
    assert len(result.executed) == 1
    assert calls == [(2, "turn_on", None)]

    rows = audit.by_trigger("trg_1")
    assert [row["kind"] for row in rows] == [
        "agent_request",
        "agent_response",
        "tool_call",
        "tool_result",
    ]
