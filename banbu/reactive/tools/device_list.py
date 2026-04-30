from __future__ import annotations

from typing import Any

from banbu.reactive.protocol import ReactiveToolCall

from .base import ReactiveToolContext, ReactiveToolRunResult, ReactiveToolSpec, audit_tool_calls
from .common import display_name, state_label


class ListRelevantDevicesTool:
    @property
    def spec(self) -> ReactiveToolSpec:
        return ReactiveToolSpec(
            name="list_relevant_devices",
            description="List cached state snapshots for devices visible to the current home context.",
            parameters={"type": "object", "properties": {}},
            safety=("Read-only. Uses SnapshotCache and never writes to IoT.",),
        )

    async def run(self, ctx: ReactiveToolContext, args: dict[str, Any]) -> ReactiveToolRunResult:
        snapshots = ctx.cache.all()
        result = {
            "ok": True,
            "count": len(snapshots),
            "devices": [
                {
                    "local_id": snap.local_id,
                    "friendly_name": snap.friendly_name,
                    "payload": snap.payload,
                }
                for snap in snapshots
            ],
        }
        tool_call = ReactiveToolCall(name=self.spec.name, args=args, result=result)
        audit_tool_calls(ctx.audit, ctx.turn, (tool_call,))

        if not snapshots:
            message = "我现在还没有家里设备的状态快照。"
        else:
            summaries = []
            for snap in snapshots[:5]:
                device = ctx.resolver.by_local_id(snap.local_id)
                name = display_name(device) if device is not None else snap.friendly_name
                if "state" in snap.payload:
                    summaries.append(f"{name}{state_label(snap.payload['state'])}")
                else:
                    summaries.append(name)
            suffix = "；还有更多设备。" if len(snapshots) > 5 else "。"
            message = "现在我能看到：" + "，".join(summaries) + suffix

        return ReactiveToolRunResult(
            final=False,
            final_message=message,
            intent=ctx.intent,
            tool_calls=(tool_call,),
        )
