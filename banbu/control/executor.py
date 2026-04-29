"""Thin async wrapper that actually pushes a translated payload to the IoT platform.

Kept separate from plane.py so the plane can hold all policy (capability
checks, idempotency, audit) and the executor only handles the wire call.
"""
from __future__ import annotations

import logging
from typing import Any

from banbu.adapters.iot_client import IoTClient

log = logging.getLogger(__name__)


class Executor:
    def __init__(self, client: IoTClient) -> None:
        self._client = client

    async def run(self, local_id: int, payload: dict[str, Any]) -> Any:
        return await self._client.control(local_id, payload)

    async def get_info(self, local_id: int) -> dict[str, Any]:
        return await self._client.get_info(local_id)
