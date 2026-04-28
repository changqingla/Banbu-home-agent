"""OpenAI tool schema (function calling) for the v1 agent.

Two tools, deliberately narrow per plan §4.3:
  - read_device_state : look up snapshot data for a device
  - execute_plan      : send an action through the control plane

The Agent never sees raw payload field names. ``action`` is a semantic verb
("turn_on", "turn_off") which the control plane translates per device.
"""
from __future__ import annotations

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_device_state",
            "description": "Read the most recent snapshot payload of a device by its local_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_id": {
                        "type": "integer",
                        "description": "Local numeric ID of the device (shown in the context block).",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of field names to return; omit for full payload.",
                    },
                },
                "required": ["local_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_plan",
            "description": (
                "Send a control action to a device. Use only the semantic actions "
                "advertised in the scene's actions_hint or supported by the device "
                "role (e.g. turn_on, turn_off). The backend rejects anything else."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_id": {"type": "integer"},
                    "action": {
                        "type": "string",
                        "description": "Semantic action verb, e.g. 'turn_on' or 'turn_off'.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional extra parameters (rarely needed).",
                    },
                },
                "required": ["local_id", "action"],
            },
        },
    },
]
