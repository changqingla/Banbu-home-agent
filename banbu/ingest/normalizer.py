"""Parse arbitrary IoT push bodies into a DeviceEvent.

The IoT platform's push schema is not documented in OpenAPI (only the
HistoryRecord shape is), so we accept multiple plausible shapes and
resolve to a managed device via the DeviceResolver. Unknown bodies
are returned as None so the webhook can log them without crashing.
"""
from __future__ import annotations

import logging
from typing import Any

from banbu.devices.resolver import DeviceResolver
from banbu.state.snapshot_cache import MISSING, SnapshotCache

from .event import DeviceEvent, FieldChange

log = logging.getLogger(__name__)


def _pick_payload(body: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("payload", "state", "data", "value"):
        v = body.get(key)
        if isinstance(v, dict):
            return v
    return None


def _resolve(body: dict[str, Any], resolver: DeviceResolver):
    for key in ("local_id", "device_id", "id"):
        v = body.get(key)
        if isinstance(v, int):
            d = resolver.by_local_id(v)
            if d:
                return d
    for key in ("ieee_address", "ieee", "mac"):
        v = body.get(key)
        if isinstance(v, str):
            d = resolver.by_ieee(v)
            if d:
                return d
    for key in ("friendly_name", "name"):
        v = body.get(key)
        if isinstance(v, str):
            d = resolver.by_name(v)
            if d:
                return d
    return None


def _diff(
    cache: SnapshotCache,
    friendly_name: str,
    new_payload: dict[str, Any],
    previous_values: dict[str, Any] | None = None,
) -> list[FieldChange]:
    changes: list[FieldChange] = []
    if previous_values is not None:
        old = previous_values
    else:
        snap = cache.get_by_name(friendly_name)
        old = snap.payload if snap else {}
    keys = set(old) | set(new_payload)
    for k in sorted(keys):
        ov = old.get(k, MISSING)
        nv = new_payload.get(k, MISSING)
        if ov is MISSING and nv is MISSING:
            continue
        if ov != nv:
            changes.append(FieldChange(field=k, old=None if ov is MISSING else ov, new=None if nv is MISSING else nv))
    return changes


def normalize_batch(
    body: Any,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    *,
    source: str = "webhook",
) -> list[DeviceEvent]:
    """Parse the v2 batch format into a list of DeviceEvents.

    Expected body shape:
        {
            "changed_at": "...",
            "reported_at": "...",
            "payload": [
                {"device_id": "sensor_01", "sequence": 101, "values": {...}},
                ...
            ]
        }
    """
    if not isinstance(body, dict):
        log.warning("ignoring non-dict batch body: %r", body)
        return []

    changed_at: str | None = body.get("changed_at")
    reported_at: str | None = body.get("reported_at")
    event_source = str(body.get("source") or source)
    items = body.get("payload", [])

    if not isinstance(items, list):
        log.warning("batch payload is not a list (got %s)", type(items).__name__)
        return []

    events: list[DeviceEvent] = []
    for item in items:
        if not isinstance(item, dict):
            log.warning("skipping non-dict batch item: %r", item)
            continue

        device_id = item.get("device_id")
        sequence = item.get("sequence")
        values = item.get("values")
        prev = item.get("previous_values") or item.get("p_values")
        if not isinstance(prev, dict):
            prev = None

        if not isinstance(values, dict):
            log.warning("batch item missing values dict: %r", _safe(item))
            continue

        device = None
        if isinstance(device_id, int):
            device = resolver.by_local_id(device_id)
        elif isinstance(device_id, str):
            try:
                device = resolver.by_local_id(int(device_id))
            except (ValueError, TypeError):
                pass
            if device is None:
                device = resolver.by_name(device_id)

        if device is None:
            log.warning("batch item device_id=%r did not resolve to a managed device", device_id)
            continue

        changes = _diff(cache, device.spec.friendly_name, values, prev)
        events.append(DeviceEvent(
            local_id=device.local_id,
            friendly_name=device.spec.friendly_name,
            ieee_address=device.ieee_address,
            payload=values,
            changes=changes,
            source=event_source,
            sequence=sequence if isinstance(sequence, int) else None,
            changed_at=changed_at,
            reported_at=reported_at,
            previous_values=prev,
        ))

    return events


def normalize(
    body: Any,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    *,
    source: str = "webhook",
) -> DeviceEvent | None:
    if not isinstance(body, dict):
        log.warning("ignoring non-dict push body: %r", body)
        return None

    device = _resolve(body, resolver)
    if device is None:
        log.warning("push body did not resolve to a managed device: %s", _safe(body))
        return None

    payload = _pick_payload(body)
    if payload is None:
        log.warning(
            "push body for %s has no recognizable payload field: %s",
            device.spec.friendly_name,
            _safe(body),
        )
        return None

    changes = _diff(cache, device.spec.friendly_name, payload)

    return DeviceEvent(
        local_id=device.local_id,
        friendly_name=device.spec.friendly_name,
        ieee_address=device.ieee_address,
        payload=payload,
        changes=changes,
        source=source,
    )


def _safe(body: Any, max_chars: int = 300) -> str:
    s = repr(body)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"
