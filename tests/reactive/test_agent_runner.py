import pytest
from types import SimpleNamespace

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.policy.access import AccessPolicy, AccessPolicyFile
from banbu.reactive.agent_runner import ReactiveAgentRunner, result_payload
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import Turn


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    async def run(self, local_id: int, payload: dict) -> None:
        self.calls.append((local_id, payload))


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="switch_entry_light",
                    room="玄关",
                    role="light_switch",
                    aliases=["玄关灯"],
                ),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="switch_kitchen_light",
                    room="厨房",
                    role="light_switch",
                    aliases=["厨房灯"],
                ),
                local_id=3,
                ieee_address="0x3",
                model="test",
                capabilities={"state"},
            ),
        ]
    )


def _entry_scene() -> Scene:
    return Scene.model_validate(
        {
            "scene_id": "entry_auto_light_v1",
            "name": "进门自动开灯",
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": "switch_entry_light",
                        "field": "payload.state",
                        "transition": "OFF->ON",
                    }
                ]
            },
            "context_devices": {"trigger": ["switch_entry_light"], "context_only": []},
            "intent": "有人进门且光线较暗时打开玄关灯",
            "actions_hint": [
                {
                    "tool": "execute_plan",
                    "args": {"device": "switch_entry_light", "action": "turn_on"},
                }
            ],
        }
    )


def _policy(user_id: str = "user_1") -> AccessPolicy:
    return AccessPolicy(
        AccessPolicyFile.model_validate(
            {
                "reactive_users": {
                    user_id: {
                        "home_id": "home_a",
                        "allowed": [
                            {
                                "device": "switch_entry_light",
                                "actions": ["turn_on", "turn_off"],
                            },
                            {
                                "device": "switch_kitchen_light",
                                "actions": ["turn_on", "turn_off"],
                            },
                        ],
                    }
                },
                "safety": {
                    "high_risk_roles": ["siren"],
                    "high_risk_actions": ["alarm_burglar", "alarm_fire"],
                    "proactive_allowed_scenes": [],
                },
            }
        )
    )


def _runner(
    tmp_path,
    *,
    responses: list[str],
    scenes: list[Scene] | None = None,
) -> tuple[ReactiveAgentRunner, FakeExecutor, AuditLog, SnapshotCache, FakeClient]:
    resolver = _resolver()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    cache = SnapshotCache(resolver)
    control = ControlPlane(executor, resolver, audit, _policy(), cache=cache)
    client = FakeClient(responses)
    runner = ReactiveAgentRunner(
        settings=Settings(llm_model="test-model"),
        resolver=resolver,
        control=control,
        audit=audit,
        cache=cache,
        scenes=scenes,
        client=client,
    )
    return runner, executor, audit, cache, client


@pytest.mark.asyncio
async def test_agent_runner_answers_greeting_without_execution(tmp_path) -> None:
    runner, executor, audit, _cache, client = _runner(
        tmp_path,
        responses=[
            '{"intent":"greeting","tool_calls":[],"final_message":"你好，我是 Banbu。"}',
        ],
    )
    turn = Turn.from_reactive("你好", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.intent == "greeting"
    assert "Banbu" in result.final_message
    assert executor.calls == []
    assert len(client.completions.requests) == 1

    rows = audit.by_trigger(turn.turn_id)
    assert [row["kind"] for row in rows] == [
        "reactive_turn",
        "reactive_agent_request",
        "reactive_agent_response",
        "reactive_intent",
        "reactive_final",
    ]
    assert rows[3]["payload"]["intent"] == "greeting"


@pytest.mark.asyncio
async def test_agent_runner_reads_device_status_snapshot(tmp_path) -> None:
    runner, executor, audit, cache, client = _runner(
        tmp_path,
        responses=[
            '{"intent":"status_query","tool_calls":[{"name":"get_device_state","args":{"local_id":2}}],"final_message":""}',
            '{"intent":"status_query","tool_calls":[],"final_message":"玄关灯现在是开着。"}',
        ],
    )
    cache.update(2, {"state": "ON", "linkquality": 255}, source="test")
    turn = Turn.from_reactive("玄关灯开着吗？", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.intent == "status_query"
    assert result.final_message == "玄关灯现在是开着。"
    assert result.snapshot is not None
    assert executor.calls == []
    assert len(client.completions.requests) == 2

    rows = audit.by_trigger(turn.turn_id)
    kinds = [row["kind"] for row in rows]
    assert kinds == [
        "reactive_turn",
        "reactive_agent_request",
        "reactive_agent_response",
        "reactive_intent",
        "reactive_tool_call",
        "reactive_tool_result",
        "reactive_agent_request",
        "reactive_agent_response",
        "reactive_final",
    ]
    assert rows[4]["payload"]["name"] == "get_device_state"


@pytest.mark.asyncio
async def test_agent_runner_executes_clear_control_request(tmp_path) -> None:
    runner, executor, _audit, _cache, client = _runner(
        tmp_path,
        responses=[
            '{"intent":"control_request","tool_calls":[{"name":"execute_plan","args":{"local_id":2,"action":"turn_on"}}],"final_message":""}',
        ],
    )
    turn = Turn.from_reactive("打开玄关灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.intent == "control_request"
    assert result.match is not None
    assert result.execution is not None
    assert result.execution.payload == {"state": "ON"}
    assert result.final_message == "已打开玄关灯。"
    assert executor.calls == [(2, {"state": "ON"})]
    assert len(client.completions.requests) == 1

    payload = result_payload(result)
    assert payload["intent"] == "control_request"
    assert payload["tool_calls"][0]["name"] == "execute_plan"


@pytest.mark.asyncio
async def test_agent_runner_scene_miss_does_not_block_direct_command(tmp_path) -> None:
    runner, executor, audit, _cache, _client = _runner(
        tmp_path,
        responses=[
            '{"intent":"control_request","tool_calls":[{"name":"execute_plan","args":{"local_id":3,"action":"turn_on"}}],"final_message":""}',
        ],
        scenes=[_entry_scene()],
    )
    turn = Turn.from_reactive("打开厨房灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.intent == "control_request"
    assert result.scene_match is None
    assert result.final_message == "已打开厨房灯。"
    assert executor.calls == [(3, {"state": "ON"})]

    rows = audit.by_trigger(turn.turn_id)
    scene_rows = [row for row in rows if row["kind"] == "reactive_scene_match"]
    assert len(scene_rows) == 1
    assert scene_rows[0]["payload"]["ok"] is False
    assert scene_rows[0]["payload"]["blocking"] is False


@pytest.mark.asyncio
async def test_agent_runner_blocks_unsafe_llm_execute_plan(tmp_path) -> None:
    runner, executor, _audit, _cache, _client = _runner(
        tmp_path,
        responses=[
            '{"intent":"control_request","tool_calls":[{"name":"execute_plan","args":{"local_id":2,"action":"turn_on"}}],"final_message":""}',
        ],
    )
    turn = Turn.from_reactive("开灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.intent == "clarification_needed"
    assert result.error_kind == "unknown_device"
    assert "设备" in result.final_message
    assert executor.calls == []
