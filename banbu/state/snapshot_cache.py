"""In-memory cache of the latest payload per managed device.

Bootstrapped from `/api/v1/devices/allinfo` at startup; updated by the
ingest pipeline (webhook + fallback poll) thereafter.

Only devices declared in `devices.yaml` are cached — events for unmanaged
devices are silently ignored at the dispatcher boundary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from banbu.adapters.iot_client import IoTClient
from banbu.devices.resolver import DeviceResolver


@dataclass
class Snapshot:
    local_id: int
    friendly_name: str
    payload: dict[str, Any]
    updated_at: float = field(default_factory=time.time)
    source: str = "boot"


_MISSING = object()


def _dig(payload: dict[str, Any], dotted: str) -> Any:
    """Resolve `payload.contact` against {'contact': ...}.

    Leading `payload.` is stripped (snapshot stores the payload object directly).
    """
    parts = dotted.split(".")
    if parts and parts[0] == "payload":
        parts = parts[1:]
    cur: Any = payload
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return _MISSING
        cur = cur[p]
    return cur


class SnapshotCache:
    def __init__(self, resolver: DeviceResolver) -> None:
        self._resolver = resolver
        self._by_local_id: dict[int, Snapshot] = {}

    async def bootstrap(self, client: IoTClient) -> None:
        all_info = await client.get_allinfo()
        for entry in all_info:
            local_id = int(entry.get("local_id", -1))
            dev = self._resolver.by_local_id(local_id)
            if dev is None:
                continue
            self._by_local_id[local_id] = Snapshot(
                local_id=local_id,
                friendly_name=dev.spec.friendly_name,
                payload=dict(entry.get("payload") or {}),
                updated_at=time.time(),
                source="bootstrap",
            )

    def update(self, local_id: int, payload: dict[str, Any], *, source: str = "event") -> Snapshot | None:
        dev = self._resolver.by_local_id(local_id)
        if dev is None:
            return None
        snap = Snapshot(
            local_id=local_id,
            friendly_name=dev.spec.friendly_name,
            payload=dict(payload),
            updated_at=time.time(),
            source=source,
        )
        self._by_local_id[local_id] = snap
        return snap

    def get(self, local_id: int) -> Snapshot | None:
        return self._by_local_id.get(local_id)

    def get_by_name(self, friendly_name: str) -> Snapshot | None:
        dev = self._resolver.by_name(friendly_name)
        if dev is None:
            return None
        return self.get(dev.local_id)

    def field(self, friendly_name: str, dotted_path: str) -> tuple[Any, float | None]:
        """Return (value, updated_at) or (_MISSING, None) if absent.

        Callers compare the returned value against `MISSING` to distinguish
        "field not present in latest payload" from "field present and equals None".
        """
        snap = self.get_by_name(friendly_name)
        if snap is None:
            return _MISSING, None
        return _dig(snap.payload, dotted_path), snap.updated_at

    def all(self) -> list[Snapshot]:
        return list(self._by_local_id.values())


MISSING = _MISSING
