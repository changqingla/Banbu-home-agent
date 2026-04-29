from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from banbu.audit.log import AuditLog
from banbu.control.plane import ControlPlane, ExecuteResult
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.turn.model import Turn

from .matcher import ReactiveMatch, ReactiveMatchError, match_device_action
from .scene_matcher import SceneMatchCandidate, SceneMatchError, select_scene_match


@dataclass(frozen=True)
class ReactiveRunResult:
    ok: bool
    turn: Turn
    scene_match: SceneMatchCandidate | None = None
    match: ReactiveMatch | None = None
    execution: ExecuteResult | None = None
    error: str | None = None
    error_kind: str | None = None


class ReactiveRunner:
    def __init__(
        self,
        *,
        resolver: DeviceResolver,
        control: ControlPlane,
        audit: AuditLog,
        scenes: list[Scene] | None = None,
    ) -> None:
        self._resolver = resolver
        self._control = control
        self._audit = audit
        self._scenes = scenes

    async def run(self, turn: Turn) -> ReactiveRunResult:
        if turn.thread_type != "reactive":
            raise ValueError(f"ReactiveRunner requires reactive turns, got {turn.thread_type!r}")
        if turn.utterance is None:
            raise ValueError("reactive turn must include utterance")

        self._audit.write(
            "reactive_turn",
            {
                "turn_id": turn.turn_id,
                "conversation_id": turn.conversation_id,
                "home_id": turn.home_id,
                "user_id": turn.user_id,
                "source": turn.source,
                "utterance": turn.utterance,
            },
            trigger_id=turn.turn_id,
        )

        scene_match: SceneMatchCandidate | None = None
        if self._scenes is not None:
            try:
                scene_match = select_scene_match(turn.utterance, self._scenes, self._resolver)
            except SceneMatchError as e:
                self._audit.write(
                    "reactive_scene_match",
                    {
                        "ok": False,
                        "kind": e.kind,
                        "error": str(e),
                        "candidates": list(e.candidates),
                    },
                    trigger_id=turn.turn_id,
                )
                return ReactiveRunResult(
                    ok=False,
                    turn=turn,
                    error=str(e),
                    error_kind=e.kind,
                )

            self._audit.write(
                "reactive_scene_match",
                {
                    "ok": True,
                    "scene_id": scene_match.scene.scene_id,
                    "score": scene_match.score,
                    "reasons": list(scene_match.reasons),
                },
                trigger_id=turn.turn_id,
                scene_id=scene_match.scene.scene_id,
            )

        try:
            match = match_device_action(turn.utterance, self._resolver)
        except ReactiveMatchError as e:
            self._audit.write(
                "reactive_match",
                {
                    "ok": False,
                    "kind": e.kind,
                    "error": str(e),
                    "candidates": list(e.candidates),
                },
                trigger_id=turn.turn_id,
            )
            return ReactiveRunResult(
                ok=False,
                turn=turn,
                scene_match=scene_match,
                error=str(e),
                error_kind=e.kind,
            )

        if scene_match is not None and match.device.spec.friendly_name not in scene_match.scene.all_referenced_devices():
            error = (
                f"matched device {match.device.spec.friendly_name!r} is not referenced by "
                f"matched scene {scene_match.scene.scene_id!r}"
            )
            self._audit.write(
                "reactive_match",
                {
                    "ok": False,
                    "kind": "device_outside_scene",
                    "error": error,
                    "local_id": match.local_id,
                    "friendly_name": match.device.spec.friendly_name,
                    "action": match.action,
                },
                trigger_id=turn.turn_id,
                scene_id=scene_match.scene.scene_id,
            )
            return ReactiveRunResult(
                ok=False,
                turn=turn,
                scene_match=scene_match,
                match=match,
                error=error,
                error_kind="device_outside_scene",
            )

        self._audit.write(
            "reactive_match",
            {
                "ok": True,
                "local_id": match.local_id,
                "friendly_name": match.device.spec.friendly_name,
                "action": match.action,
                "action_terms": list(match.action_terms),
                "device_reasons": list(match.device_reasons),
            },
            trigger_id=turn.turn_id,
            scene_id=scene_match.scene.scene_id if scene_match is not None else None,
        )

        execution = await self._control.execute(
            match.local_id,
            match.action,
            None,
            trigger_id=turn.turn_id,
            scene_id=scene_match.scene.scene_id if scene_match is not None else None,
        )
        return ReactiveRunResult(
            ok=execution.ok,
            turn=turn,
            scene_match=scene_match,
            match=match,
            execution=execution,
            error=execution.error,
            error_kind=None if execution.ok else "execute_failed",
        )


def result_payload(result: ReactiveRunResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": result.ok,
        "turn_id": result.turn.turn_id,
        "conversation_id": result.turn.conversation_id,
    }
    if result.scene_match is not None:
        payload["scene_match"] = {
            "scene_id": result.scene_match.scene.scene_id,
            "score": result.scene_match.score,
            "reasons": list(result.scene_match.reasons),
        }
    if result.match is not None:
        payload["match"] = {
            "local_id": result.match.local_id,
            "friendly_name": result.match.device.spec.friendly_name,
            "action": result.match.action,
            "device_reasons": list(result.match.device_reasons),
        }
    if result.execution is not None:
        payload["execution"] = {
            "ok": result.execution.ok,
            "local_id": result.execution.local_id,
            "action": result.execution.action,
            "payload": result.execution.payload,
            "error": result.execution.error,
            "deduped": result.execution.deduped,
        }
    if result.error is not None:
        payload["error"] = result.error
        payload["error_kind"] = result.error_kind
    return payload
