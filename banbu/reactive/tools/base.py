from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from banbu.audit.log import AuditLog
from banbu.control.plane import ControlPlane, ExecuteResult
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import Snapshot, SnapshotCache
from banbu.turn.model import Turn

from banbu.reactive.matcher import ReactiveDeviceMention, ReactiveMatch
from banbu.reactive.protocol import ReactiveIntent, ReactiveToolCall
from banbu.reactive.scene_matcher import SceneMatchCandidate


@dataclass(frozen=True)
class ReactiveToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    safety: tuple[str, ...] = ()


@dataclass
class ReactiveToolContext:
    turn: Turn
    intent: ReactiveIntent
    resolver: DeviceResolver
    cache: SnapshotCache
    control: ControlPlane
    audit: AuditLog
    scenes: list[Scene] | None = None


@dataclass(frozen=True)
class ReactiveToolRunResult:
    final: bool
    final_message: str
    ok: bool = True
    intent: ReactiveIntent | None = None
    scene_match: SceneMatchCandidate | None = None
    match: ReactiveMatch | None = None
    device_mention: ReactiveDeviceMention | None = None
    snapshot: Snapshot | None = None
    execution: ExecuteResult | None = None
    tool_calls: tuple[ReactiveToolCall, ...] = ()
    error: str | None = None
    error_kind: str | None = None


class ReactiveTool(Protocol):
    @property
    def spec(self) -> ReactiveToolSpec:
        ...

    async def run(self, ctx: ReactiveToolContext, args: dict[str, Any]) -> ReactiveToolRunResult:
        ...


def audit_tool_calls(
    audit: AuditLog,
    turn: Turn,
    tool_calls: tuple[ReactiveToolCall, ...],
    scene_match: SceneMatchCandidate | None = None,
) -> None:
    for call in tool_calls:
        audit.write(
            "reactive_tool_call",
            {"name": call.name, "args": call.args},
            trigger_id=turn.turn_id,
            scene_id=scene_match.scene.scene_id if scene_match is not None else None,
        )
        audit.write(
            "reactive_tool_result",
            {"name": call.name, "result": call.result},
            trigger_id=turn.turn_id,
            scene_id=scene_match.scene.scene_id if scene_match is not None else None,
        )
