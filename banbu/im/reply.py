from __future__ import annotations

from banbu.reactive.runner import ReactiveRunResult

ACTION_LABELS = {
    "turn_on": "打开",
    "turn_off": "关闭",
}


def render_reactive_reply(result: ReactiveRunResult) -> str:
    if result.ok and result.match is not None:
        action = ACTION_LABELS.get(result.match.action, result.match.action)
        name = result.match.device.spec.aliases[0] if result.match.device.spec.aliases else result.match.device.spec.friendly_name
        if result.execution is not None and result.execution.deduped:
            return f"已收到，{name} 的{action}指令刚刚执行过。"
        return f"已{action}{name}。"

    if result.error:
        return f"没能完成这次请求：{result.error}"
    return "没能完成这次请求。"
