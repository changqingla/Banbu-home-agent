"""Pick the relevant slice of state for one Turn (plan §9.3).

Selection rules (v1, proactive only):
  1. Determine the target scene from the Turn's scene_id.
  2. Take exactly the devices the scene references — trigger devices and
     context_only devices.
  3. Skip everything else (other rooms, unrelated devices, full home snapshot).
"""
from __future__ import annotations

from dataclasses import dataclass

from banbu.devices.definition import ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.state.feedback import FeedbackEntry, FeedbackStore
from banbu.state.snapshot_cache import Snapshot, SnapshotCache
from banbu.turn.model import Turn


@dataclass
class SelectedContext:
    turn: Turn
    scene: Scene
    devices: list[ResolvedDevice]
    snapshots: dict[str, Snapshot]
    feedback: list[FeedbackEntry]


def select(
    turn: Turn,
    scene: Scene,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    feedback_store: FeedbackStore | None = None,
) -> SelectedContext:
    devices: list[ResolvedDevice] = []
    snapshots: dict[str, Snapshot] = {}

    seen: set[str] = set()
    for name in scene.all_referenced_devices():
        if name in seen:
            continue
        seen.add(name)
        dev = resolver.by_name(name)
        if dev is None:
            continue
        devices.append(dev)
        snap = cache.get_by_name(name)
        if snap is not None:
            snapshots[name] = snap

    feedback = []
    if feedback_store is not None and turn.scene_id is not None:
        feedback = feedback_store.recent(turn.home_id, turn.scene_id)

    return SelectedContext(
        turn=turn,
        scene=scene,
        devices=devices,
        snapshots=snapshots,
        feedback=feedback,
    )
