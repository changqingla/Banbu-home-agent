"""Thin wrapper over ``contextpilot.optimize``.

Returns ready-to-send OpenAI messages with documents reordered for KV-cache
prefix sharing across turns of the same conversation.
"""
from __future__ import annotations

import logging
from typing import Any

import contextpilot as cp

from banbu.agent.prompts import TURN_QUERY_TEMPLATE

log = logging.getLogger(__name__)


def optimize(
    blocks: list[str],
    *,
    conversation_id: str,
    query: str | None = None,
) -> list[dict[str, Any]]:
    q = query or TURN_QUERY_TEMPLATE
    try:
        messages = cp.optimize(blocks, q, conversation_id=conversation_id)
    except Exception as e:
        log.warning("contextpilot.optimize failed (%s); falling back to plain assembly", e)
        joined = "\n\n".join(f"[{i+1}] {b}" for i, b in enumerate(blocks))
        messages = [
            {"role": "system", "content": joined},
            {"role": "user", "content": q},
        ]
    return messages
