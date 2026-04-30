from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from banbu.audit.log import AuditLog
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.state.snapshot_cache import SnapshotCache


class RecordingExecutor:
    def __init__(
        self,
        *,
        refreshed_payload: dict[str, Any] | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []
        self._refreshed_payload = refreshed_payload or {"state": "ON"}
        self._refresh_error = refresh_error

    async def run(self, local_id: int, payload: dict[str, Any]) -> None:
        self.calls.append((local_id, payload))

    async def get_info(self, local_id: int) -> dict[str, Any]:
        if self._refresh_error is not None:
            raise self._refresh_error
        return {
            "id": 17937,
            "ieee_address": "0xlight",
            "payload": dict(self._refreshed_payload),
            "timestamp": "2026-04-29 01:43:01",
        }


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="entry_light",
                    role="light_switch",
                    care_fields=["state"],
                    actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
                ),
                local_id=12,
                ieee_address="0xlight",
                model="TS011F",
                capabilities={"state"},
            )
        ]
    )


def _audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "audit.sqlite")


@pytest.mark.asyncio
async def test_successful_execute_refreshes_target_snapshot(tmp_path: Path) -> None:
    resolver = _resolver()
    cache = SnapshotCache(resolver)
    cache.update(12, {"state": "OFF"}, source="test")
    audit = _audit(tmp_path)
    executor = RecordingExecutor(refreshed_payload={"state": "ON", "linkquality": 236})
    control = ControlPlane(executor, resolver, audit, cache=cache)

    result = await control.execute(
        12,
        "turn_on",
        trigger_id="trg_refresh_ok",
        scene_id="entry_auto_light_v1",
    )

    assert result.ok is True
    assert executor.calls == [(12, {"state": "ON"})]
    snap = cache.get(12)
    assert snap is not None
    assert snap.payload == {"state": "ON", "linkquality": 236}
    assert snap.source == "control_refresh"

    rows = audit.by_trigger("trg_refresh_ok")
    refresh = next(row for row in rows if row["kind"] == "snapshot_refresh")
    assert refresh["payload"] == {
        "local_id": 12,
        "ok": True,
        "payload": {"state": "ON", "linkquality": 236},
    }


@pytest.mark.asyncio
async def test_refresh_failure_keeps_successful_execute_result(tmp_path: Path) -> None:
    resolver = _resolver()
    cache = SnapshotCache(resolver)
    cache.update(12, {"state": "OFF"}, source="test")
    audit = _audit(tmp_path)
    executor = RecordingExecutor(refresh_error=RuntimeError("device info timeout"))
    control = ControlPlane(executor, resolver, audit, cache=cache)

    result = await control.execute(
        12,
        "turn_on",
        trigger_id="trg_refresh_fail",
        scene_id="entry_auto_light_v1",
    )

    assert result.ok is True
    assert cache.get(12).payload == {"state": "OFF"}

    rows = audit.by_trigger("trg_refresh_fail")
    refresh = next(row for row in rows if row["kind"] == "snapshot_refresh")
    execute_result = next(row for row in rows if row["kind"] == "execute_result")
    assert refresh["payload"] == {
        "local_id": 12,
        "ok": False,
        "error": "device info timeout",
    }
    assert execute_result["payload"]["ok"] is True
