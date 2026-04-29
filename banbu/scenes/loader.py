"""Glob, validate, and cross-check scene yaml files.

Failure modes:
- File missing the schema fields → pydantic raises, scene rejected (whole boot aborts).
- Scene references a device not in devices.yaml → reject.
- Scene references a field outside the device's exposed capabilities → reject.

We never half-load: any error means the boot must fail loudly so the user
fixes the yaml before the system processes events.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from banbu.devices.resolver import DeviceResolver

from .definition import DurationTrigger, Scene, Trigger, VisionTrigger, WindowedAllTrigger

log = logging.getLogger(__name__)


class SceneLoadError(RuntimeError):
    pass


def _strip_payload_prefix(field: str) -> str:
    return field[len("payload."):] if field.startswith("payload.") else field


def _validate_against_devices(scene: Scene, resolver: DeviceResolver) -> list[str]:
    errors: list[str] = []

    for name in scene.all_referenced_devices():
        if resolver.by_name(name) is None:
            errors.append(f"unknown device {name!r} (not in devices.yaml)")

    if isinstance(scene.trigger, Trigger):
        for step in scene.trigger.steps:
            dev = resolver.by_name(step.device)
            if dev is None:
                continue
            leaf = _strip_payload_prefix(step.field)
            if leaf not in dev.capabilities:
                errors.append(
                    f"trigger step on {step.device}: field {step.field!r} not in capabilities {sorted(dev.capabilities)}"
                )
    elif isinstance(scene.trigger, WindowedAllTrigger):
        for condition in scene.trigger.conditions:
            dev = resolver.by_name(condition.device)
            if dev is None:
                continue
            leaf = _strip_payload_prefix(condition.field)
            if leaf not in dev.capabilities:
                errors.append(
                    f"windowed condition on {condition.device}: field {condition.field!r} not in capabilities {sorted(dev.capabilities)}"
                )
    elif isinstance(scene.trigger, DurationTrigger):
        condition = scene.trigger.condition
        dev = resolver.by_name(condition.device)
        if dev is not None:
            leaf = _strip_payload_prefix(condition.field)
            if leaf not in dev.capabilities:
                errors.append(
                    f"duration condition on {condition.device}: field {condition.field!r} not in capabilities {sorted(dev.capabilities)}"
                )
    elif isinstance(scene.trigger, VisionTrigger):
        dev = resolver.by_name(scene.trigger.device)
        if dev is not None:
            for field in (
                scene.trigger.field,
                scene.trigger.confidence_field,
                scene.trigger.detected_field,
                scene.trigger.frame_id_field,
            ):
                leaf = _strip_payload_prefix(field)
                if leaf not in dev.capabilities:
                    errors.append(
                        f"vision trigger on {scene.trigger.device}: field {field!r} not in capabilities {sorted(dev.capabilities)}"
                    )

    for pre in scene.preconditions:
        dev = resolver.by_name(pre.device)
        if dev is None:
            continue
        leaf = _strip_payload_prefix(pre.field)
        if leaf not in dev.capabilities:
            errors.append(
                f"precondition on {pre.device}: field {pre.field!r} not in capabilities {sorted(dev.capabilities)}"
            )

    return errors


def load_scenes(scenes_dir: Path, resolver: DeviceResolver) -> list[Scene]:
    if not scenes_dir.exists():
        log.warning("scenes dir %s does not exist; loading zero scenes", scenes_dir)
        return []

    files = sorted(scenes_dir.glob("*.yaml")) + sorted(scenes_dir.glob("*.yml"))
    scenes: list[Scene] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    for path in files:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"{path.name}: yaml parse error ({e})")
            continue

        if not raw:
            errors.append(f"{path.name}: empty file")
            continue

        try:
            scene = Scene.model_validate(raw)
        except ValidationError as e:
            errors.append(f"{path.name}: schema error\n{e}")
            continue

        if scene.scene_id in seen_ids:
            errors.append(f"{path.name}: duplicate scene_id {scene.scene_id!r}")
            continue

        device_errors = _validate_against_devices(scene, resolver)
        if device_errors:
            joined = "\n    - ".join(device_errors)
            errors.append(f"{path.name} ({scene.scene_id}):\n    - {joined}")
            continue

        scenes.append(scene)
        seen_ids.add(scene.scene_id)
        if isinstance(scene.trigger, Trigger):
            trigger_count = len(scene.trigger.steps)
        elif isinstance(scene.trigger, WindowedAllTrigger):
            trigger_count = len(scene.trigger.conditions)
        elif isinstance(scene.trigger, DurationTrigger):
            trigger_count = 1
        else:
            trigger_count = 1
        log.info("loaded scene %s from %s (%d trigger steps, %d preconditions)",
                 scene.scene_id, path.name, trigger_count, len(scene.preconditions))

    if errors:
        joined = "\n  - ".join(errors)
        raise SceneLoadError(f"scene validation errors:\n  - {joined}")

    log.info("loaded %d scenes total from %s", len(scenes), scenes_dir)
    return scenes
