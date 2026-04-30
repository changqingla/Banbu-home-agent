from __future__ import annotations

import json
from typing import Any

from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import Turn

from .protocol import ReactiveAgentDecision, ReactiveToolCall
from .tools.common import device_context

SYSTEM_PROMPT_TEMPLATE = """\
You are Banbu, a bounded smart-home reactive agent for IM user turns.

Return exactly one JSON object. Do not use markdown.

Schema:
{{
  "intent": "greeting|help|status_query|control_request|clarification_needed|unsupported",
  "tool_calls": [
    {{"name": "<available_tool_name>", "args": {{"...": "..."}}}}
  ],
  "final_message": "Short Chinese reply if no tool is needed or after using provided tool results."
}}

Available tools:
{tools_json}

Rules:
- You may call at most one tool per response.
- Use read-only tools for status questions.
- Use execute_plan only when the user explicitly asks for a concrete device action.
- If the device/action is ambiguous, ask a clarification question and do not call execute_plan.
- Do not invent local_id, action, device names, or state values. Use only the device context.
- All device writes are executed by backend policy; you only propose execute_plan.
- Reply in concise Chinese.
"""


def render_system_prompt(tool_specs: list[dict[str, Any]]) -> str:
    tools_json = json.dumps(tool_specs, ensure_ascii=False, indent=2, sort_keys=True)
    return SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json)


def render_user_payload(
    *,
    turn: Turn,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    scenes: list[Scene] | None,
) -> str:
    payload = {
        "home_id": turn.home_id,
        "user_id": turn.user_id,
        "source": turn.source,
        "utterance": turn.utterance,
        "devices": device_context(resolver, cache),
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "name": scene.name,
                "intent": scene.intent,
                "referenced_devices": list(scene.all_referenced_devices()),
            }
            for scene in scenes or []
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def initial_messages(
    *,
    turn: Turn,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    scenes: list[Scene] | None,
    tool_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": render_system_prompt(tool_specs)},
        {
            "role": "user",
            "content": render_user_payload(turn=turn, resolver=resolver, cache=cache, scenes=scenes),
        },
    ]


def tool_followup_messages(
    *,
    messages: list[dict[str, Any]],
    decision: ReactiveAgentDecision,
    tool_calls: tuple[ReactiveToolCall, ...],
) -> list[dict[str, Any]]:
    payload = {
        "previous_decision": {
            "intent": decision.intent,
            "tool_calls": [
                {"name": call.name, "args": call.args}
                for call in decision.tool_calls
            ],
            "final_message": decision.final_message,
        },
        "tool_results": [
            {"name": call.name, "args": call.args, "result": call.result}
            for call in tool_calls
        ],
        "instruction": "Use the tool result to produce the final user-facing Chinese reply. Do not call another tool.",
    }
    return [
        *messages,
        {"role": "assistant", "content": decision.raw},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
