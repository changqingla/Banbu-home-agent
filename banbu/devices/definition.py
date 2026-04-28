from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DeviceSpec(BaseModel):
    friendly_name: str
    role: str
    care_fields: list[str] = Field(default_factory=list)
    actions: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DevicesFile(BaseModel):
    devices: list[DeviceSpec]


class ResolvedDevice(BaseModel):
    spec: DeviceSpec
    local_id: int
    ieee_address: str
    model: str
    capabilities: set[str]

    model_config = {"arbitrary_types_allowed": True}


def effective_actions(spec: DeviceSpec) -> dict[str, dict[str, Any]]:
    return dict(spec.actions)
