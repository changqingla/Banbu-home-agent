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
  6. Refreshes the target snapshot from /devices/info.
  7. Writes audit rows.
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
from banbu.state.snapshot_cache import SnapshotCache

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


@dataclass
class _ConflictClaim:
    local_id: int
    payload: dict[str, Any]
    scene_id: str | None
    priority: int
    expires_at: float


class ControlPlane:
    def __init__(
        self,
        executor: Executor,
        resolver: DeviceResolver,
        cache: SnapshotCache,
        audit: AuditLog,
        *,
        idempotency_window_seconds: float = 5.0,
        conflict_window_seconds: float = 2.0,
        scene_priorities: dict[str, int] | None = None,
    ) -> None:
        self._executor = executor
        self._resolver = resolver
        self._cache = cache
        self._audit = audit
        self._window = idempotency_window_seconds
        self._recent: dict[str, float] = {}
        self._conflict_window = conflict_window_seconds
        self._scene_priorities = dict(scene_priorities or {})
        self._claims: dict[int, _ConflictClaim] = {}

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

    def _scene_priority(self, scene_id: str | None) -> int:
        if scene_id is None:
            return 0
        return self._scene_priorities.get(scene_id, 0)

    def _prune_conflict_claims(self, now: float) -> None:
        for local_id, claim in list(self._claims.items()):
            if now >= claim.expires_at:
                self._claims.pop(local_id, None)

    def _payload_conflicts(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        return any(key in right and right[key] != value for key, value in left.items())

    def _claim_conflict(
        self,
        *,
        local_id: int,
        payload: dict[str, Any],
        scene_id: str | None,
        trigger_id: str | None,
    ) -> str | None:
        now = time.time()
        self._prune_conflict_claims(now)

        priority = self._scene_priority(scene_id)
        existing = self._claims.get(local_id)
        if existing is not None and self._payload_conflicts(existing.payload, payload):
            if priority <= existing.priority:
                error = (
                    f"conflicting action for local_id={local_id}: scene {scene_id!r} "
                    f"priority={priority} conflicts with scene {existing.scene_id!r} "
                    f"priority={existing.priority}"
                )
                self._audit.write(
                    "conflict_reject",
                    {
                        "local_id": local_id,
                        "scene_id": scene_id,
                        "priority": priority,
                        "payload": payload,
                        "existing_scene_id": existing.scene_id,
                        "existing_priority": existing.priority,
                        "existing_payload": existing.payload,
                    },
                    trigger_id=trigger_id,
                    scene_id=scene_id,
                )
                return error

            self._audit.write(
                "conflict_override",
                {
                    "local_id": local_id,
                    "scene_id": scene_id,
                    "priority": priority,
                    "payload": payload,
                    "overridden_scene_id": existing.scene_id,
                    "overridden_priority": existing.priority,
                    "overridden_payload": existing.payload,
                },
                trigger_id=trigger_id,
                scene_id=scene_id,
            )

        self._claims[local_id] = _ConflictClaim(
            local_id=local_id,
            payload=dict(payload),
            scene_id=scene_id,
            priority=priority,
            expires_at=now + self._conflict_window,
        )
        return None

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

        conflict_error = self._claim_conflict(
            local_id=local_id,
            payload=payload,
            scene_id=scene_id,
            trigger_id=trigger_id,
        )
        if conflict_error is not None:
            log.warning("control: conflict rejected local_id=%d action=%s: %s", local_id, action, conflict_error)
            self._audit.write(
                "execute_result",
                {
                    "local_id": local_id,
                    "action": action,
                    "payload": payload,
                    "ok": False,
                    "error": conflict_error,
                },
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            return ExecuteResult(ok=False, local_id=local_id, action=action, payload=payload, error=conflict_error)

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
            claim = self._claims.get(local_id)
            if claim is not None and claim.scene_id == scene_id and claim.payload == payload:
                self._claims.pop(local_id, None)
            self._audit.write(
                "execute_result",
                {"local_id": local_id, "action": action, "ok": False, "error": str(e)},
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            return ExecuteResult(ok=False, local_id=local_id, action=action, payload=payload, error=str(e))

        log.info("control: executed local_id=%d action=%s payload=%s", local_id, action, payload)
        await self._refresh_snapshot(local_id, trigger_id=trigger_id, scene_id=scene_id)
        self._audit.write(
            "execute_result",
            {"local_id": local_id, "action": action, "payload": payload, "ok": True},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )
        return ExecuteResult(ok=True, local_id=local_id, action=action, payload=payload)

    async def _refresh_snapshot(
        self,
        local_id: int,
        *,
        trigger_id: str | None,
        scene_id: str | None,
    ) -> None:
        try:
            info = await self._executor.get_info(local_id)
            payload = info.get("payload")
            if not isinstance(payload, dict):
                raise ControlError(f"/devices/info local_id={local_id} returned no payload object")
        except Exception as e:
            log.warning("control: snapshot refresh failed for local_id=%d: %s", local_id, e)
            self._audit.write(
                "snapshot_refresh",
                {"local_id": local_id, "ok": False, "error": str(e)},
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            return

        snap = self._cache.update(local_id, payload, source="control_refresh")
        if snap is None:
            log.warning("control: snapshot refresh ignored unmanaged local_id=%d", local_id)
            self._audit.write(
                "snapshot_refresh",
                {"local_id": local_id, "ok": False, "error": "unmanaged local_id"},
                trigger_id=trigger_id,
                scene_id=scene_id,
            )
            return

        log.info("control: snapshot refreshed local_id=%d source=%s", local_id, snap.source)
        self._audit.write(
            "snapshot_refresh",
            {"local_id": local_id, "ok": True, "payload": payload},
            trigger_id=trigger_id,
            scene_id=scene_id,
        )
