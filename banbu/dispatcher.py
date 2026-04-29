"""Route DeviceEvents to scene runtimes via the reverse index.

Per architecture doc 8.1: the reverse index identifies *candidate* scenes
for an incoming event; the runtime decides whether the scene actually
hits. role=context_only events are no-ops here because the snapshot
cache has already been updated upstream (in webhook + poller).
"""
from __future__ import annotations

import logging

from banbu.ingest.event import DeviceEvent
from banbu.scenes.definition import Scene
from banbu.scenes.reverse_index import ReverseIndex
from banbu.scenes.runtime.base import OnHit, SceneRuntime
from banbu.scenes.runtime.duration import DurationSceneRuntime
from banbu.scenes.runtime.edge import EdgeSceneRuntime
from banbu.scenes.runtime.sequential import SequentialSceneRuntime
from banbu.scenes.runtime.vision_match import VisionMatchSceneRuntime
from banbu.scenes.runtime.windowed_all import WindowedAllSceneRuntime
from banbu.state.snapshot_cache import SnapshotCache

log = logging.getLogger(__name__)


class Dispatcher:
    def __init__(
        self,
        scenes: list[Scene],
        reverse_index: ReverseIndex,
        cache: SnapshotCache,
        *,
        home_id: str,
        on_hit: OnHit | None = None,
    ) -> None:
        self._reverse_index = reverse_index
        self._runtimes: dict[str, SceneRuntime] = {}
        self._priorities: dict[str, int] = {scene.scene_id: scene.policy.priority for scene in scenes}
        for scene in scenes:
            if scene.kind == "sequential":
                self._runtimes[scene.scene_id] = SequentialSceneRuntime(
                    scene, cache, home_id=home_id, on_hit=on_hit
                )
            elif scene.kind == "edge_triggered":
                self._runtimes[scene.scene_id] = EdgeSceneRuntime(
                    scene, cache, home_id=home_id, on_hit=on_hit
                )
            elif scene.kind == "windowed_all":
                self._runtimes[scene.scene_id] = WindowedAllSceneRuntime(
                    scene, cache, home_id=home_id, on_hit=on_hit
                )
            elif scene.kind == "duration_triggered":
                self._runtimes[scene.scene_id] = DurationSceneRuntime(
                    scene, cache, home_id=home_id, on_hit=on_hit
                )
            elif scene.kind == "vision_match":
                self._runtimes[scene.scene_id] = VisionMatchSceneRuntime(
                    scene, cache, home_id=home_id, on_hit=on_hit
                )
        unsupported = [s.scene_id for s in scenes if s.scene_id not in self._runtimes]
        if unsupported:
            log.warning("scenes with unsupported kind ignored in v1: %s", unsupported)

    def runtime(self, scene_id: str) -> SceneRuntime | None:
        return self._runtimes.get(scene_id)

    def mark_executed(
        self,
        scene_id: str,
        *,
        success: bool,
        cooldown_seconds: float | None = None,
    ) -> None:
        runtime = self._runtimes.get(scene_id)
        if runtime is None:
            return
        state = getattr(runtime, "state", None)
        if state is None:
            return
        state.clear_inflight()
        if success and cooldown_seconds:
            state.set_cooldown(cooldown_seconds)
            log.info("scene %s: cooldown set for %.0fs after successful execute", scene_id, cooldown_seconds)
        else:
            log.info(
                "scene %s: inflight cleared (success=%s, cooldown not written)",
                scene_id, success,
            )

    def on_event(self, event: DeviceEvent) -> None:
        for change in event.changes:
            field_key = change.field if change.field.startswith("payload.") else f"payload.{change.field}"
            entries = self._reverse_index.lookup(event.friendly_name, field_key)
            if not entries:
                continue
            trigger_scene_ids = sorted(
                (scene_id for scene_id, role in entries if role == "trigger"),
                key=lambda scene_id: (-self._priorities.get(scene_id, 0), scene_id),
            )
            for scene_id in trigger_scene_ids:
                runtime = self._runtimes.get(scene_id)
                if runtime is None:
                    continue
                runtime.on_event(event, change)

    def on_tick(self) -> None:
        for runtime in self._runtimes.values():
            on_tick = getattr(runtime, "on_tick", None)
            if on_tick is not None:
                on_tick()
