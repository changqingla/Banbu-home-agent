from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldChange:
    field: str
    old: Any
    new: Any


@dataclass
class DeviceEvent:
    local_id: int
    friendly_name: str
    ieee_address: str
    payload: dict[str, Any]
    changes: list[FieldChange] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "webhook"
    sequence: int | None = None
    changed_at: str | None = None
    reported_at: str | None = None
    previous_values: dict[str, Any] | None = None
