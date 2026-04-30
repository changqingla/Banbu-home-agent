from __future__ import annotations

from banbu.reactive.agent_runner import ReactiveAgentResult


def render_reactive_reply(result: ReactiveAgentResult) -> str:
    return result.final_message
