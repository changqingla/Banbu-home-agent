"""FastAPI router accepting batch device events from the IoT platform."""
from __future__ import annotations

import logging
from typing import Callable

from fastapi import APIRouter, Request

from banbu.devices.resolver import DeviceResolver
from banbu.state.snapshot_cache import SnapshotCache

from .event import DeviceEvent
from .normalizer import normalize_batch

log = logging.getLogger(__name__)

EventHandler = Callable[[DeviceEvent], None]


def make_router(
    *,
    path: str,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    on_event: EventHandler | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post(path)
    async def receive(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception as e:
            log.warning("webhook: invalid JSON body (%s)", e)
            return {"ok": True, "ignored": True, "reason": "invalid_json"}

        log.info("webhook: body=%s", body)

        events = normalize_batch(body, resolver, cache, source="webhook")

        total_changes = 0
        for event in events:
            for ch in event.changes:
                log.info(
                    "EVENT %s/%s local_id=%d seq=%s %s: %r -> %r",
                    event.source,
                    event.friendly_name,
                    event.local_id,
                    event.sequence,
                    ch.field,
                    ch.old,
                    ch.new,
                )
            if not event.changes:
                log.info(
                    "EVENT %s/%s local_id=%d seq=%s (no diff) payload=%s",
                    event.source,
                    event.friendly_name,
                    event.local_id,
                    event.sequence,
                    event.payload,
                )
            total_changes += len(event.changes)
            cache.update(event.local_id, event.payload, source=event.source)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    log.exception("on_event handler raised (continuing)")

        return {"ok": True, "processed": len(events), "changes": total_changes}

    return router
