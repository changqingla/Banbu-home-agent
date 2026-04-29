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
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="entry_light",
                    role="light_switch",
                    actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
                ),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state"},
            )
        ]
    )


def test_translate_uses_device_actions_and_validates_capabilities(tmp_path) -> None:
    plane = ControlPlane(FakeExecutor(), _resolver(), AuditLog(tmp_path / "audit.sqlite"))

    assert plane.translate(2, "turn_on", None) == {"state": "ON"}

    with pytest.raises(Exception, match="not supported"):
        plane.translate(2, "brightness_high", None)


@pytest.mark.asyncio
async def test_execute_calls_executor_and_dedupes_recent_duplicate(tmp_path) -> None:
    executor = FakeExecutor()
    audit = AuditLog(tmp_path / "audit.sqlite")
    plane = ControlPlane(executor, _resolver(), audit, idempotency_window_seconds=60)

    first = await plane.execute(2, "turn_on", trigger_id="trg_1", scene_id="scene_a")
    second = await plane.execute(2, "turn_on", trigger_id="trg_1", scene_id="scene_a")

    assert first.ok is True
    assert second.ok is True
    assert second.deduped is True
    assert executor.calls == [(2, {"state": "ON"})]

    rows = audit.by_trigger("trg_1")
    assert [row["kind"] for row in rows] == ["execute", "execute_result"]
