from __future__ import annotations

from typing import Any

from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import TriggerStep, WILDCARD


def _match_value(spec: Any, actual: Any) -> bool:
    if spec is WILDCARD:
        return True
    return spec == actual


def matches_step(step: TriggerStep, event: DeviceEvent, change: FieldChange) -> bool:
    if event.friendly_name != step.device:
        return False
    norm_field = step.field if step.field.startswith("payload.") else f"payload.{step.field}"
    leaf = norm_field[len("payload."):]
    if change.field != leaf:
        return False
    return _match_value(step.old_value, change.old) and _match_value(step.new_value, change.new)


def summarize_change(event: DeviceEvent, change: FieldChange) -> str:
    return f"{event.friendly_name}.{change.field}: {change.old!r}->{change.new!r}"
