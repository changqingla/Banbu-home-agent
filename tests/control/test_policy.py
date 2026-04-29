import pytest

from banbu.audit.log import AuditLog
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.policy.access import AccessPolicy, AccessPolicyFile


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
                    role="light_switch",
                    actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
                ),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="entry_siren",
                    role="siren",
                    actions={"alarm_fire": {"warning": {"mode": "fire"}}},
                ),
                local_id=6,
                ieee_address="0x6",
                model="test",
                capabilities={"warning"},
            ),
        ]
    )


def _policy() -> AccessPolicy:
    return AccessPolicy(
        AccessPolicyFile.model_validate(
            {
                "reactive_users": {
                    "user_1": {
                        "home_id": "home_a",
                        "allowed": [
                            {
                                "device": "switch_entry_light",
                                "actions": ["turn_on", "turn_off"],
                            },
                            {
                                "device": "entry_siren",
                                "actions": ["alarm_fire"],
                            },
                        ],
                    }
                },
                "safety": {
                    "high_risk_roles": ["siren"],
                    "high_risk_actions": ["alarm_fire"],
                    "proactive_allowed_scenes": ["safety_smoke_then_gas_v1"],
                },
            }
        )
    )


@pytest.mark.asyncio
async def test_control_denies_unauthorized_reactive_user_before_execution(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, _resolver(), audit, _policy())

    result = await control.execute(
        2,
        "turn_on",
        trigger_id="turn_1",
        scene_id="entry_auto_light_v1",
        actor="reactive",
        home_id="home_a",
        user_id="stranger",
    )

    assert result.ok is False
    assert executor.calls == []

    rows = audit.by_trigger("turn_1")
    assert [row["kind"] for row in rows] == ["policy_denied", "execute_result"]
    assert rows[0]["payload"]["user_id"] == "stranger"
    assert rows[1]["payload"]["ok"] is False


@pytest.mark.asyncio
async def test_control_allows_authorized_reactive_user(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, _resolver(), audit, _policy())

    result = await control.execute(
        2,
        "turn_on",
        trigger_id="turn_1",
        scene_id="entry_auto_light_v1",
        actor="reactive",
        home_id="home_a",
        user_id="user_1",
    )

    assert result.ok is True
    assert executor.calls == [(2, {"state": "ON"})]


@pytest.mark.asyncio
async def test_control_denies_reactive_high_risk_action_even_if_allowlisted(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, _resolver(), audit, _policy())

    result = await control.execute(
        6,
        "alarm_fire",
        trigger_id="turn_1",
        scene_id="safety_smoke_then_gas_v1",
        actor="reactive",
        home_id="home_a",
        user_id="user_1",
    )

    assert result.ok is False
    assert "high-risk" in (result.error or "")
    assert executor.calls == []


@pytest.mark.asyncio
async def test_control_allows_only_approved_proactive_high_risk_scenes(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, _resolver(), audit, _policy())

    denied = await control.execute(
        6,
        "alarm_fire",
        trigger_id="trg_1",
        scene_id="entry_auto_light_v1",
        actor="proactive",
        home_id="home_a",
    )
    allowed = await control.execute(
        6,
        "alarm_fire",
        trigger_id="trg_2",
        scene_id="safety_smoke_then_gas_v1",
        actor="proactive",
        home_id="home_a",
    )

    assert denied.ok is False
    assert allowed.ok is True
    assert executor.calls == [(6, {"warning": {"mode": "fire"}})]
