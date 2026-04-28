"""Backend hard boundary between Agent and the device platform (plan §6.3 / §6.5).

Agent only sees semantic actions (turn_on / turn_off / set_temperature). The
control plane:
  1. Resolves the device.
  2. Translates the action to a device-shaped payload via role-default tables
     and per-device overrides in devices.yaml.
  3. Verifies every key in the resulting payload exists in the device's
     /exposes capability set.
  4. Idempotency: hash(trigger_id + local_id + payload) deduped for 5s.
  5. Calls executor → IoT platform.
  6. Writes audit rows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from banbu.adapters.iot_client import IoTError
from banbu.audit.log import AuditLog
from banbu.devices.definition import effective_actions
from banbu.devices.resolver import DeviceResolver

from .executor import Executor

log = logging.getLogger(__name__)


class ControlError(RuntimeError):
    pass


@dataclass
class ExecuteResult:
    ok: bool
    local_id: int
    action: str
    payload: dict[str, Any]
    error: str | None = None
    deduped: bool = False


class ControlPlane:
    def __init__(
        self,
        executor: Executor,
        resolver: DeviceResolver,
        audit: AuditLog,
        *,
        idempotency_window_seconds: float = 5.0,
    ) -> None:
        self._executor = executor
        self._resolver = resolver
        self._audit = audit
        self._window = idempotency_window_seconds
        self._recent: dict[str, float] = {}

    def translate(self, local_id: int, action: str, params: dict[str, Any] | None) -> dict[str, Any]:
        dev = self._resolver.by_local_id(local_id)
        if dev is None:
            raise ControlError(f"unknown local_id={local_id}")

        actions = effective_actions(dev.spec)
        template = actions.get(action)
        if template is None:
            raise ControlError(
                f"action {action!r} not supported for {dev.spec.friendly_name} "
                f"(role={dev.spec.role}, available={sorted(actions)})"
            )

        payload: dict[str, Any] = dict(template)
        if params:
            payload.update(params)

        for key in payload:
            if key not in dev.capabilities:
                raise ControlError(
                    f"action {action!r} produced field {key!r} not in capabilities "
                    f"{sorted(dev.capabilities)}"
                )
        return payload

    def _dedupe_key(self, trigger_id: str | None, local_id: int, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, sort_keys=True)
        h = hashlib.sha1(f"{trigger_id or '-'}|{local_id}|{body}".encode()).hexdigest()[:16]
        return h

    def _is_recent_duplicate(self, key: str) -> bool:
        now = time.time()
        for k, t in list(self._recent.items()):
            if now - t > self._window:
                self._recent.pop(k, None)
        if key in self._recent:
            return True
        self._recent[key] = now
        return False

    async def execute(
        self,
        local_id: int,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        trigger_id: str | None = None,
        scene_id: str | None = None,
    ) -> ExecuteResult:
        try:
            payload = self.translate(local_id, action, params)
        except ControlError as e:
            log.warning("control: translate rejected %s/%s: %s", local_id, action, e)
            self._audit.write(
                "execute_result",
                {"local_id": local_id, "action": action, "ok": False, "error": str(e)},
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            return ExecuteResult(ok=False, local_id=local_id, action=action, payload={}, error=str(e))

        key = self._dedupe_key(trigger_id, local_id, payload)
        if self._is_recent_duplicate(key):
            log.info("control: deduped local_id=%d action=%s payload=%s", local_id, action, payload)
            return ExecuteResult(ok=True, local_id=local_id, action=action, payload=payload, deduped=True)

        self._audit.write(
            "execute",
            {"local_id": local_id, "action": action, "payload": payload},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )
        try:
            await self._executor.run(local_id, payload)
        except IoTError as e:
            log.warning("control: IoT error on local_id=%d: %s", local_id, e)
            self._audit.write(
                "execute_result",
                {"local_id": local_id, "action": action, "ok": False, "error": str(e)},
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            return ExecuteResult(ok=False, local_id=local_id, action=action, payload=payload, error=str(e))

        log.info("control: executed local_id=%d action=%s payload=%s", local_id, action, payload)
        self._audit.write(
            "execute_result",
            {"local_id": local_id, "action": action, "payload": payload, "ok": True},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )
        return ExecuteResult(ok=True, local_id=local_id, action=action, payload=payload)
