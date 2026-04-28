"""Fallback poll: catches events the IoT platform failed to push.

Every BANBU_FALLBACK_POLL_SECONDS, fetch /devices/allinfo and synthesize
DeviceEvents for any managed device whose payload differs from the
snapshot cache. Synthetic events carry source="poll" so downstream code
can distinguish them from real-time pushes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from banbu.adapters.iot_client import IoTClient
from banbu.devices.resolver import DeviceResolver
from banbu.state.snapshot_cache import MISSING, SnapshotCache

from .event import DeviceEvent, FieldChange

log = logging.getLogger(__name__)

EventHandler = Callable[[DeviceEvent], None]


def _diff(old: dict, new: dict) -> list[FieldChange]:
    changes: list[FieldChange] = []
    for k in sorted(set(old) | set(new)):
        ov = old.get(k, MISSING)
        nv = new.get(k, MISSING)
        if ov is MISSING and nv is MISSING:
            continue
        if ov != nv:
            changes.append(FieldChange(field=k, old=None if ov is MISSING else ov, new=None if nv is MISSING else nv))
    return changes


class FallbackPoller:
    def __init__(
        self,
        client: IoTClient,
        resolver: DeviceResolver,
        cache: SnapshotCache,
        *,
        interval_seconds: int,
        on_event: EventHandler | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._cache = cache
        self._interval = interval_seconds
        self._on_event = on_event
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="banbu-fallback-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval + 2)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _run(self) -> None:
        log.info("fallback poller started (interval=%ds)", self._interval)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await self._tick()
            except Exception:
                log.exception("fallback poll tick failed")
        log.info("fallback poller stopped")

    async def _tick(self) -> None:
        all_info = await self._client.get_allinfo()
        for entry in all_info:
            local_id = int(entry.get("local_id", -1))
            dev = self._resolver.by_local_id(local_id)
            if dev is None:
                continue
            new_payload = dict(entry.get("payload") or {})
            snap = self._cache.get(local_id)
            old_payload = snap.payload if snap else {}
            changes = _diff(old_payload, new_payload)
            if not changes:
                continue
            for ch in changes:
                log.info(
                    "EVENT poll/%s local_id=%d %s: %r -> %r",
                    dev.spec.friendly_name,
                    local_id,
                    ch.field,
                    ch.old,
                    ch.new,
                )
            self._cache.update(local_id, new_payload, source="poll")
            if self._on_event is not None:
                try:
                    self._on_event(
                        DeviceEvent(
                            local_id=local_id,
                            friendly_name=dev.spec.friendly_name,
                            ieee_address=dev.ieee_address,
                            payload=new_payload,
                            changes=changes,
                            source="poll",
                        )
                    )
                except Exception:
                    log.exception("on_event handler raised (continuing)")
