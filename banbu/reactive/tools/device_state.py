from __future__ import annotations

from typing import Any

from banbu.reactive.matcher import ReactiveDeviceMention, ReactiveMatchError, match_device_mention
from banbu.reactive.protocol import ReactiveToolCall

from .base import ReactiveToolContext, ReactiveToolRunResult, ReactiveToolSpec, audit_tool_calls
from .common import display_name, snapshot_summary


class GetDeviceStateTool:
    @property
    def spec(self) -> ReactiveToolSpec:
        return ReactiveToolSpec(
            name="get_device_state",
            description="Read the latest cached state snapshot for one device.",
            parameters={
                "type": "object",
                "required": ["local_id"],
                "properties": {
                    "local_id": {
                        "type": "integer",
                        "description": "Device local_id from the provided device context.",
                    }
                },
            },
            safety=("Read-only. Uses SnapshotCache and never writes to IoT.",),
        )

    async def run(self, ctx: ReactiveToolContext, args: dict[str, Any]) -> ReactiveToolRunResult:
        mention = self._safe_device_mention(ctx, args)
        snapshot = ctx.cache.get(mention.local_id) if mention is not None else None
        result = {
            "ok": mention is not None,
            "local_id": mention.local_id if mention is not None else args.get("local_id"),
            "friendly_name": mention.device.spec.friendly_name if mention is not None else None,
            "display_name": display_name(mention.device) if mention is not None else None,
            "payload": snapshot.payload if snapshot is not None else None,
            "error": None if mention is not None else "could not safely identify the requested device",
        }
        tool_call = ReactiveToolCall(name=self.spec.name, args=args, result=result)
        audit_tool_calls(ctx.audit, ctx.turn, (tool_call,))
        return ReactiveToolRunResult(
            final=False,
            final_message=snapshot_summary(mention.device, snapshot) if mention is not None else "我不确定你要查哪个设备。",
            intent=ctx.intent,
            device_mention=mention,
            snapshot=snapshot,
            tool_calls=(tool_call,),
            error=result["error"],
            error_kind=None if mention is not None else "unknown_device",
        )

    def _safe_device_mention(self, ctx: ReactiveToolContext, args: dict[str, Any]) -> ReactiveDeviceMention | None:
        try:
            mention = match_device_mention(ctx.turn.utterance or "", ctx.resolver)
        except ReactiveMatchError:
            local_id = args.get("local_id")
            if local_id is None:
                return None
            device = ctx.resolver.by_local_id(int(local_id))
            if device is None:
                return None
            return ReactiveDeviceMention(device=device, device_reasons=("llm_selected_local_id",))

        requested_local_id = args.get("local_id")
        if requested_local_id is not None and int(requested_local_id) != mention.local_id:
            return None
        return mention
