from __future__ import annotations

import logging
import time
from typing import Any

from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene, Trigger
from banbu.state.snapshot_cache import MISSING, SnapshotCache
from banbu.turn.model import ProactiveTrigger

from .base import OnHit, SceneRuntime
from .conditions import PreconditionFailed, check_precondition
from .lifecycle import SceneState
from .transitions import matches_step, summarize_change

log = logging.getLogger(__name__)


class EdgeSceneRuntime(SceneRuntime):
    def __init__(
        self,
        scene: Scene,
        cache: SnapshotCache,
        *,
        home_id: str,
        on_hit: OnHit | None = None,
    ) -> None:
        super().__init__(scene, on_hit=on_hit)
        if scene.kind != "edge_triggered":
            raise ValueError(f"EdgeSceneRuntime got kind={scene.kind!r}")
        if not isinstance(scene.trigger, Trigger):
            raise ValueError("edge runtime requires trigger.steps")
        if len(scene.trigger.steps) != 1:
            raise ValueError("edge_triggered scenes require exactly one trigger step")
        self._cache = cache
        self._home_id = home_id
        self.state = SceneState()

    def on_event(self, event: DeviceEvent, change: FieldChange) -> None:
        scene = self.scene
        assert isinstance(scene.trigger, Trigger)
        st = self.state
        now = time.time()

        if st.is_in_cooldown(now):
            log.info(
                "scene %s: cooldown active (%.1fs left), dropping %s.%s",
                scene.scene_id,
                st.cooldown_until - now,
                event.friendly_name,
                change.field,
            )
            return
        if st.is_inflight(now):
            log.info(
                "scene %s: inflight active (%.1fs left), dropping %s.%s",
                scene.scene_id,
                st.inflight_until - now,
                event.friendly_name,
                change.field,
            )
            return

        step = scene.trigger.steps[0]
        if not matches_step(step, event, change):
            return

        self._evaluate_and_emit(event, change)

    def _evaluate_and_emit(self, event: DeviceEvent, change: FieldChange) -> None:
        scene = self.scene
        st = self.state

        try:
            results: list[str] = []
            all_pass = True
            for pre in scene.preconditions:
                ok, summary = check_precondition(pre, self._cache)
                results.append(summary)
                if not ok:
                    all_pass = False
        except PreconditionFailed as e:
            log.warning("scene %s: precondition failed (%s) - abort", scene.scene_id, e)
            return

        if not all_pass:
            log.info(
                "scene %s: preconditions rejected - no HIT. results: %s",
                scene.scene_id,
                "; ".join(results),
            )
            return

        facts = self._collect_facts()
        trigger = ProactiveTrigger(
            scene_id=scene.scene_id,
            home_id=self._home_id,
            facts=facts,
            source_event_summaries=[summarize_change(event, change)],
        )
        st.set_inflight(scene.policy.inflight_seconds)
        log.info(
            "Scene HIT %s trigger_id=%s inflight_until=%.0f facts=%s",
            scene.scene_id,
            trigger.trigger_id,
            st.inflight_until,
            facts,
        )

        if self._on_hit is not None:
            try:
                self._on_hit(trigger)
            except Exception:
                log.exception("on_hit handler raised (continuing)")

    def _collect_facts(self) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        for name in self.scene.trigger_devices():
            snap = self._cache.get_by_name(name)
            if snap is not None:
                facts[name] = dict(snap.payload)
        for pre in self.scene.preconditions:
            value, _ = self._cache.field(pre.device, pre.field)
            facts.setdefault(pre.device, {})[pre.field] = None if value is MISSING else value
        return facts
