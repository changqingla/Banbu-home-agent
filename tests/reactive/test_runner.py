import pytest

from banbu.audit.log import AuditLog
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.policy.access import AccessPolicy, AccessPolicyFile
from banbu.reactive.runner import ReactiveRunner
from banbu.scenes.definition import Scene
from banbu.turn.model import Turn


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    async def run(self, local_id: int, payload: dict) -> None:
        self.calls.append((local_id, payload))


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
            )
        ]
    )


def _resolver_with_kitchen_light() -> DeviceResolver:
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
                            }
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


def test_turn_from_reactive_sets_thread_identity() -> None:
    turn = Turn.from_reactive("打开玄关灯", home_id="home_a", user_id="user_1")

    assert turn.thread_type == "reactive"
    assert turn.conversation_id == "home_a_user_1"
    assert turn.home_id == "home_a"
    assert turn.user_id == "user_1"
    assert turn.utterance == "打开玄关灯"
    assert turn.scene_id is None
    assert turn.trigger is None


@pytest.mark.asyncio
async def test_runner_executes_matched_turn_through_control_plane(tmp_path) -> None:
    resolver = _resolver()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, resolver, audit, _policy())
    runner = ReactiveRunner(resolver=resolver, control=control, audit=audit)
    turn = Turn.from_reactive("打开玄关灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.match is not None
    assert result.match.local_id == 2
    assert result.execution is not None
    assert result.execution.payload == {"state": "ON"}
    assert executor.calls == [(2, {"state": "ON"})]

    rows = audit.by_trigger(turn.turn_id)
    assert [row["kind"] for row in rows] == [
        "reactive_turn",
        "reactive_match",
        "execute",
        "execute_result",
    ]
    assert rows[1]["payload"]["ok"] is True
    assert rows[1]["payload"]["action"] == "turn_on"


@pytest.mark.asyncio
async def test_runner_rejects_unknown_device_without_execution(tmp_path) -> None:
    resolver = _resolver()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, resolver, audit, _policy())
    runner = ReactiveRunner(resolver=resolver, control=control, audit=audit)
    turn = Turn.from_reactive("打开厨房灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is False
    assert result.error_kind == "unknown_device"
    assert executor.calls == []

    rows = audit.by_trigger(turn.turn_id)
    assert [row["kind"] for row in rows] == ["reactive_turn", "reactive_match"]
    assert rows[1]["payload"]["ok"] is False
    assert rows[1]["payload"]["kind"] == "unknown_device"


@pytest.mark.asyncio
async def test_runner_with_scene_guard_executes_matched_scene_device(tmp_path) -> None:
    resolver = _resolver()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, resolver, audit, _policy())
    runner = ReactiveRunner(resolver=resolver, control=control, audit=audit, scenes=[_entry_scene()])
    turn = Turn.from_reactive("打开玄关灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is True
    assert result.scene_match is not None
    assert result.scene_match.scene.scene_id == "entry_auto_light_v1"
    assert executor.calls == [(2, {"state": "ON"})]

    rows = audit.by_trigger(turn.turn_id)
    assert [row["kind"] for row in rows] == [
        "reactive_turn",
        "reactive_scene_match",
        "reactive_match",
        "execute",
        "execute_result",
    ]
    assert rows[1]["scene_id"] == "entry_auto_light_v1"
    assert rows[3]["scene_id"] == "entry_auto_light_v1"


@pytest.mark.asyncio
async def test_runner_with_scene_guard_rejects_no_match_without_execution(tmp_path) -> None:
    resolver = _resolver_with_kitchen_light()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, resolver, audit, _policy())
    runner = ReactiveRunner(resolver=resolver, control=control, audit=audit, scenes=[_entry_scene()])
    turn = Turn.from_reactive("打开厨房灯", home_id="home_a", user_id="user_1")

    result = await runner.run(turn)

    assert result.ok is False
    assert result.error_kind == "no_scene_match"
    assert executor.calls == []

    rows = audit.by_trigger(turn.turn_id)
    assert [row["kind"] for row in rows] == ["reactive_turn", "reactive_scene_match"]
    assert rows[1]["payload"]["ok"] is False
    assert rows[1]["payload"]["kind"] == "no_scene_match"
