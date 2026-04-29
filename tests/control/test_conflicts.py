import pytest

from banbu.audit.log import AuditLog
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    async def run(self, local_id: int, payload: dict) -> None:
        self.calls.append((local_id, payload))


def _resolver() -> DeviceResolver:
    device = ResolvedDevice(
        spec=DeviceSpec(
            friendly_name="switch_entry_light",
            role="light_switch",
            actions={
                "turn_on": {"state": "ON"},
                "turn_off": {"state": "OFF"},
                "brightness_mid": {"brightness": 127},
            },
        ),
        local_id=2,
        ieee_address="0x2",
        model="test",
        capabilities={"state", "brightness"},
    )
    return DeviceResolver([device])


@pytest.mark.asyncio
async def test_lower_priority_conflicting_action_is_rejected(tmp_path) -> None:
    executor = FakeExecutor()
    audit = AuditLog(tmp_path / "audit.sqlite")
    control = ControlPlane(
        executor,
        _resolver(),
        audit,
        scene_priorities={"high_scene": 10, "low_scene": 1},
        conflict_window_seconds=60,
    )

    first = await control.execute(2, "turn_on", trigger_id="t1", scene_id="high_scene")
    second = await control.execute(2, "turn_off", trigger_id="t2", scene_id="low_scene")

    assert first.ok is True
    assert second.ok is False
    assert "conflicting action" in (second.error or "")
    assert executor.calls == [(2, {"state": "ON"})]

    rows = audit.by_trigger("t2")
    assert [row["kind"] for row in rows] == ["conflict_reject", "execute_result"]
    assert rows[0]["payload"]["existing_scene_id"] == "high_scene"
    assert rows[1]["payload"]["ok"] is False


@pytest.mark.asyncio
async def test_higher_priority_conflicting_action_overrides_claim(tmp_path) -> None:
    executor = FakeExecutor()
    audit = AuditLog(tmp_path / "audit.sqlite")
    control = ControlPlane(
        executor,
        _resolver(),
        audit,
        scene_priorities={"high_scene": 10, "low_scene": 1},
        conflict_window_seconds=60,
    )

    first = await control.execute(2, "turn_off", trigger_id="t1", scene_id="low_scene")
    second = await control.execute(2, "turn_on", trigger_id="t2", scene_id="high_scene")

    assert first.ok is True
    assert second.ok is True
    assert executor.calls == [(2, {"state": "OFF"}), (2, {"state": "ON"})]

    rows = audit.by_trigger("t2")
    assert [row["kind"] for row in rows] == ["conflict_override", "execute", "execute_result"]
    assert rows[0]["payload"]["overridden_scene_id"] == "low_scene"


@pytest.mark.asyncio
async def test_disjoint_payload_fields_do_not_conflict(tmp_path) -> None:
    executor = FakeExecutor()
    audit = AuditLog(tmp_path / "audit.sqlite")
    control = ControlPlane(
        executor,
        _resolver(),
        audit,
        scene_priorities={"scene_a": 5, "scene_b": 5},
        conflict_window_seconds=60,
    )

    first = await control.execute(2, "turn_on", trigger_id="t1", scene_id="scene_a")
    second = await control.execute(2, "brightness_mid", trigger_id="t2", scene_id="scene_b")

    assert first.ok is True
    assert second.ok is True
    assert executor.calls == [(2, {"state": "ON"}), (2, {"brightness": 127})]

    rows = audit.by_trigger("t2")
    assert [row["kind"] for row in rows] == ["execute", "execute_result"]
