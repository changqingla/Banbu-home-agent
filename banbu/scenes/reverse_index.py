"""(friendly_name, field) -> [(scene_id, role)] lookup table.

Built once at boot and read on every device event. Field keys use the
dotted form (``payload.contact``) so they match the field strings used
by trigger steps and DeviceEvent diffs.

Roles:
- ``trigger``      : the event must be routed to this scene's runtime.
- ``context_only`` : the event only updates the snapshot cache; no runtime work.
"""
from __future__ import annotations

from typing import Literal

from banbu.devices.resolver import DeviceResolver

from .definition import DurationTrigger, Scene, Trigger, VisionTrigger, WindowedAllTrigger

Role = Literal["trigger", "context_only"]


class ReverseIndex:
    def __init__(self) -> None:
        self._idx: dict[tuple[str, str], list[tuple[str, Role]]] = {}

    def add(self, device: str, field: str, scene_id: str, role: Role) -> None:
        key = (device, field)
        bucket = self._idx.setdefault(key, [])
        if (scene_id, role) not in bucket:
            bucket.append((scene_id, role))

    def lookup(self, device: str, field: str) -> list[tuple[str, Role]]:
        return list(self._idx.get((device, field), ()))

    def all(self) -> dict[tuple[str, str], list[tuple[str, Role]]]:
        return {k: list(v) for k, v in self._idx.items()}

    def __len__(self) -> int:
        return sum(len(v) for v in self._idx.values())


def _normalize(field: str) -> str:
    return field if field.startswith("payload.") else f"payload.{field}"


def build_reverse_index(scenes: list[Scene], resolver: DeviceResolver) -> ReverseIndex:
    idx = ReverseIndex()
    for scene in scenes:
        if isinstance(scene.trigger, Trigger):
            for step in scene.trigger.steps:
                idx.add(step.device, _normalize(step.field), scene.scene_id, "trigger")
        elif isinstance(scene.trigger, WindowedAllTrigger):
            for condition in scene.trigger.conditions:
                idx.add(condition.device, _normalize(condition.field), scene.scene_id, "trigger")
        elif isinstance(scene.trigger, DurationTrigger):
            condition = scene.trigger.condition
            idx.add(condition.device, _normalize(condition.field), scene.scene_id, "trigger")
        elif isinstance(scene.trigger, VisionTrigger):
            for field in (
                scene.trigger.field,
                scene.trigger.confidence_field,
                scene.trigger.detected_field,
                scene.trigger.frame_id_field,
            ):
                idx.add(scene.trigger.device, _normalize(field), scene.scene_id, "trigger")
        for name in scene.context_devices.context_only:
            dev = resolver.by_name(name)
            if dev is None:
                continue
            for cf in dev.spec.care_fields:
                idx.add(name, _normalize(cf), scene.scene_id, "context_only")
    return idx
