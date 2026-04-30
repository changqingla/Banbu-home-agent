from __future__ import annotations

from typing import Any

from banbu.reactive.matcher import ReactiveMatch, ReactiveMatchError, match_device_action
from banbu.reactive.protocol import ReactiveToolCall
from banbu.reactive.scene_matcher import SceneMatchCandidate, SceneMatchError, select_scene_match

from .base import ReactiveToolContext, ReactiveToolRunResult, ReactiveToolSpec, audit_tool_calls
from .common import display_name


class ExecutePlanTool:
    @property
    def spec(self) -> ReactiveToolSpec:
        return ReactiveToolSpec(
            name="execute_plan",
            description="Execute a semantic smart-home action through ControlPlane.",
            parameters={
                "type": "object",
                "required": ["local_id", "action"],
                "properties": {
                    "local_id": {"type": "integer"},
                    "action": {
                        "type": "string",
                        "description": "Semantic action, e.g. turn_on or turn_off.",
                    },
                    "params": {
                        "type": ["object", "null"],
                        "description": "Optional action parameters. Usually null.",
                    },
                },
            },
            safety=(
                "Writes are only proposals from the LLM.",
                "Backend must match the same device and action from the user utterance before execution.",
                "All writes go through ControlPlane policy, capability, idempotency, and conflict checks.",
            ),
        )

    async def run(self, ctx: ReactiveToolContext, args: dict[str, Any]) -> ReactiveToolRunResult:
        safe_match, guard_error, guard_kind = self._safe_control_match(ctx, args)
        if safe_match is None:
            tool_call = ReactiveToolCall(
                name=self.spec.name,
                args=args,
                result={"ok": False, "error": guard_error, "guard": guard_kind},
            )
            audit_tool_calls(ctx.audit, ctx.turn, (tool_call,))
            return ReactiveToolRunResult(
                final=True,
                final_message=self._clarification_message(guard_kind, guard_error),
                intent="clarification_needed",
                tool_calls=(tool_call,),
                error=guard_error,
                error_kind=guard_kind,
            )

        scene_match = self._select_scene_for_control(ctx, safe_match)
        execution = await ctx.control.execute(
            safe_match.local_id,
            safe_match.action,
            args.get("params"),
            trigger_id=ctx.turn.turn_id,
            scene_id=scene_match.scene.scene_id if scene_match is not None else None,
            actor="reactive",
            home_id=ctx.turn.home_id,
            user_id=ctx.turn.user_id,
        )
        tool_args = {
            "local_id": safe_match.local_id,
            "action": safe_match.action,
            "params": args.get("params"),
        }
        tool_call = ReactiveToolCall(
            name=self.spec.name,
            args=tool_args,
            result={
                "ok": execution.ok,
                "local_id": execution.local_id,
                "action": execution.action,
                "payload": execution.payload,
                "error": execution.error,
                "deduped": execution.deduped,
            },
        )
        audit_tool_calls(ctx.audit, ctx.turn, (tool_call,), scene_match)

        if execution.ok:
            action_label = self._action_label(safe_match.action)
            name = display_name(safe_match.device)
            if execution.deduped:
                final_message = f"已收到，{name} 的{action_label}指令刚刚执行过。"
            else:
                final_message = f"已{action_label}{name}。"
            error = None
            error_kind = None
        else:
            final_message = f"没能完成这次请求：{execution.error or '执行失败'}"
            error = execution.error
            error_kind = "execute_failed"

        return ReactiveToolRunResult(
            final=True,
            ok=execution.ok,
            intent=ctx.intent,
            final_message=final_message,
            scene_match=scene_match,
            match=safe_match,
            execution=execution,
            tool_calls=(tool_call,),
            error=error,
            error_kind=error_kind,
        )

    def _safe_control_match(
        self,
        ctx: ReactiveToolContext,
        args: dict[str, Any],
    ) -> tuple[ReactiveMatch | None, str | None, str | None]:
        try:
            match = match_device_action(ctx.turn.utterance or "", ctx.resolver)
        except ReactiveMatchError as e:
            return None, str(e), e.kind

        requested_local_id = args.get("local_id")
        requested_action = args.get("action")
        if requested_local_id is not None and int(requested_local_id) != match.local_id:
            return None, "LLM selected a device that was not explicitly matched from the utterance", "unsafe_device"
        if requested_action is not None and str(requested_action) != match.action:
            return None, "LLM selected an action that was not explicitly matched from the utterance", "unsafe_action"
        return match, None, None

    def _action_label(self, action: str) -> str:
        if action == "turn_on":
            return "打开"
        if action == "turn_off":
            return "关闭"
        return action

    def _select_scene_for_control(self, ctx: ReactiveToolContext, match: ReactiveMatch) -> SceneMatchCandidate | None:
        if ctx.scenes is None:
            return None
        try:
            scene_match = select_scene_match(ctx.turn.utterance or "", ctx.scenes, ctx.resolver)
        except SceneMatchError as e:
            ctx.audit.write(
                "reactive_scene_match",
                {
                    "ok": False,
                    "kind": e.kind,
                    "error": str(e),
                    "candidates": list(e.candidates),
                    "blocking": False,
                },
                trigger_id=ctx.turn.turn_id,
            )
            return None

        referenced = scene_match.scene.all_referenced_devices()
        if match.device.spec.friendly_name not in referenced:
            ctx.audit.write(
                "reactive_scene_match",
                {
                    "ok": False,
                    "kind": "device_outside_scene",
                    "scene_id": scene_match.scene.scene_id,
                    "local_id": match.local_id,
                    "friendly_name": match.device.spec.friendly_name,
                    "blocking": False,
                },
                trigger_id=ctx.turn.turn_id,
                scene_id=scene_match.scene.scene_id,
            )
            return None

        ctx.audit.write(
            "reactive_scene_match",
            {
                "ok": True,
                "scene_id": scene_match.scene.scene_id,
                "score": scene_match.score,
                "reasons": list(scene_match.reasons),
                "blocking": False,
            },
            trigger_id=ctx.turn.turn_id,
            scene_id=scene_match.scene.scene_id,
        )
        return scene_match

    def _clarification_message(self, error_kind: str | None, error: str | None) -> str:
        if error_kind == "ambiguous_device":
            return "我不确定你指的是哪个设备，请说得更具体一点。"
        if error_kind in {"unknown_device", "unsafe_device"}:
            return "我没找到你说的设备。你可以换成设备别名，比如“玄关灯”。"
        if error_kind in {"unknown_action", "unsafe_action"}:
            return "我还没判断出你要执行什么动作。你可以说“打开玄关灯”或“关闭玄关灯”。"
        if error_kind == "unsupported_action":
            return "这个设备目前不支持你说的动作。"
        return error or "我还没理解这句话对应的家庭请求。"
