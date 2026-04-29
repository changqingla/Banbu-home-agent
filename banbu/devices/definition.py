from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


ROLE_DEFAULT_ACTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "light_switch": {
        "turn_on": {"state": "ON"},
        "turn_off": {"state": "OFF"},
    },
    "smart_plug": {
        "turn_on": {"state": "ON"},
        "turn_off": {"state": "OFF"},
    },
    "color_temp_light": {
        "turn_on": {"state": "ON"},
        "turn_off": {"state": "OFF"},
        "brightness_high": {"brightness": 254},
        "brightness_mid": {"brightness": 127},
        "brightness_low": {"brightness": 50},
        "color_temp_cool": {"color_temp": 250},
        "color_temp_neutral": {"color_temp": 370},
        "color_temp_warm": {"color_temp": 454},
    },
}


class DeviceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: str
    room: str | None = None
    role: str
    aliases: list[str] = Field(default_factory=list)
    care_fields: list[str] = Field(default_factory=list)
    actions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    virtual: bool = False
    local_id: int | None = None
    ieee_address: str | None = None
    model: str = "virtual"
    capabilities: list[str] | None = None


class DevicesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    devices: list[DeviceSpec]


class ResolvedDevice(BaseModel):
    spec: DeviceSpec
    local_id: int
    ieee_address: str
    model: str
    capabilities: set[str]

    model_config = {"arbitrary_types_allowed": True}


def effective_actions(spec: DeviceSpec) -> dict[str, dict[str, Any]]:
    actions = deepcopy(ROLE_DEFAULT_ACTIONS.get(spec.role, {}))
    for name, payload in spec.actions.items():
        actions[name] = deepcopy(payload)
    return actions
