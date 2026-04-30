from __future__ import annotations

from typing import Any

from banbu.devices.definition import ResolvedDevice, effective_actions
from banbu.devices.resolver import DeviceResolver
from banbu.state.snapshot_cache import Snapshot, SnapshotCache


def display_name(device: ResolvedDevice) -> str:
    return device.spec.aliases[0] if device.spec.aliases else device.spec.friendly_name


def state_label(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.upper()
        if normalized == "ON":
            return "开着"
        if normalized == "OFF":
            return "关着"
    if isinstance(value, bool):
        return "开着" if value else "关着"
    return repr(value)


def snapshot_summary(device: ResolvedDevice, snapshot: Snapshot | None) -> str:
    name = display_name(device)
    if snapshot is None:
        return f"我现在还没有{name}的状态快照。"

    payload = snapshot.payload
    if "state" in payload:
        return f"{name}现在是{state_label(payload['state'])}。"

    care_fields = [field for field in device.spec.care_fields if field in payload]
    if care_fields:
        facts = "，".join(f"{field}={payload[field]!r}" for field in care_fields[:4])
        return f"{name}当前状态：{facts}。"

    if payload:
        facts = "，".join(f"{key}={value!r}" for key, value in list(sorted(payload.items()))[:4])
        return f"{name}当前状态：{facts}。"

    return f"我有{name}的设备记录，但当前快照是空的。"


def device_context(resolver: DeviceResolver, cache: SnapshotCache) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for device in resolver.all():
        snapshot = cache.get(device.local_id)
        devices.append(
            {
                "local_id": device.local_id,
                "friendly_name": device.spec.friendly_name,
                "display_name": display_name(device),
                "room": device.spec.room,
                "role": device.spec.role,
                "aliases": list(device.spec.aliases),
                "actions": sorted(effective_actions(device.spec)),
                "care_fields": list(device.spec.care_fields),
                "snapshot": snapshot.payload if snapshot is not None else None,
            }
        )
    return devices
