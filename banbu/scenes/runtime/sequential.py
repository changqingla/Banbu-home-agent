"""Sequential trigger state machine + precondition evaluation.

Lifecycle (see plan §4.2.1 for the full table):

  [event arrives]
       │
       ├── cooldown active?     drop
       ├── inflight active?     drop
       ├── window expired?      reset cursor → 0
       ├── matches steps[cursor]?
       │       yes → advance cursor
       │             ├── if cursor == len(steps):
       │             │       evaluate preconditions
       │             │            pass → emit ProactiveTrigger,
       │             │                   set inflight_until,
       │             │                   reset cursor
       │             │            fail → reset cursor (no inflight, no cooldown)
       │       no  → if cursor>0 and event matches steps[0]:
       │                   restart at step 1 (we just matched step 0 fresh)
"""
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


class SequentialSceneRuntime(SceneRuntime):
    def __init__(
        self,
        scene: Scene,
        cache: SnapshotCache,
        *,
        home_id: str,
        on_hit: OnHit | None = None,
    ) -> None:
        super().__init__(scene, on_hit=on_hit)
        if scene.kind != "sequential":
            raise ValueError(f"SequentialSceneRuntime got kind={scene.kind!r}")
        if not isinstance(scene.trigger, Trigger):
            raise ValueError("sequential runtime requires trigger.steps")
        self._cache = cache
        self._home_id = home_id
        self.state = SceneState()
        self._recent_events: list[str] = []
        self._recent_event_ids: list[str] = []

    def on_event(self, event: DeviceEvent, change: FieldChange) -> None:
        scene = self.scene
        assert isinstance(scene.trigger, Trigger)
        steps = scene.trigger.steps
        st = self.state
        now = time.time()

        if st.is_in_cooldown(now):
            log.info(
                "scene %s: cooldown active (%.1fs left), dropping %s.%s",
                scene.scene_id, st.cooldown_until - now,
                event.friendly_name, change.field,
            )
            return
        if st.is_inflight(now):
            log.info(
                "scene %s: inflight active (%.1fs left), dropping %s.%s",
                scene.scene_id, st.inflight_until - now,
                event.friendly_name, change.field,
            )
            return

        if st.cursor > 0 and st.last_step_at is not None:
            window = steps[st.cursor].within_seconds
            if window is not None and (now - st.last_step_at) > window:
                log.info(
                    "scene %s: window expired (%.1fs > %.1fs), resetting cursor",
                    scene.scene_id, now - st.last_step_at, window,
                )
                st.reset_cursor()

        target = steps[st.cursor]
        if matches_step(target, event, change):
            self._note_event(event, change)
            st.last_step_at = now
            st.cursor += 1
            log.info(
                "scene %s: step %d/%d matched on %s.%s (%r->%r)",
                scene.scene_id, st.cursor, len(steps),
                event.friendly_name, change.field, change.old, change.new,
            )
            if st.cursor >= len(steps):
                self._evaluate_and_emit()
            return

        if st.cursor > 0 and matches_step(steps[0], event, change):
            log.info(
                "scene %s: restart from step 0 on %s.%s",
                scene.scene_id, event.friendly_name, change.field,
            )
            self._recent_events.clear()
            self._recent_event_ids.clear()
            self._note_event(event, change)
            st.cursor = 1
            st.last_step_at = now

    def _note_event(self, event: DeviceEvent, change: FieldChange) -> None:
        self._recent_events.append(summarize_change(event, change))
        self._recent_event_ids.append(event.event_id)
        if len(self._recent_events) > 10:
            self._recent_events = self._recent_events[-10:]
            self._recent_event_ids = self._recent_event_ids[-10:]

    def _evaluate_and_emit(self) -> None:
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
            log.warning("scene %s: precondition failed (%s) — abort", scene.scene_id, e)
            st.reset_cursor()
            self._recent_events.clear()
            self._recent_event_ids.clear()
            return

        if not all_pass:
            log.info(
                "scene %s: preconditions rejected — no HIT. results: %s",
                scene.scene_id, "; ".join(results),
            )
            st.reset_cursor()
            self._recent_events.clear()
            self._recent_event_ids.clear()
            return

        facts = self._collect_facts()
        trigger = ProactiveTrigger(
            scene_id=scene.scene_id,
            home_id=self._home_id,
            facts=facts,
            source_event_summaries=list(self._recent_events),
            source_event_ids=list(self._recent_event_ids),
        )
        st.set_inflight(scene.policy.inflight_seconds)
        log.info(
            "Scene HIT %s trigger_id=%s inflight_until=%.0f facts=%s",
            scene.scene_id, trigger.trigger_id, st.inflight_until, facts,
        )
        st.reset_cursor()
        self._recent_events.clear()
        self._recent_event_ids.clear()

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
