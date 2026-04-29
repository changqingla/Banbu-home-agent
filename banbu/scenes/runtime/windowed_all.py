from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene, WindowedAllTrigger
from banbu.state.snapshot_cache import MISSING, SnapshotCache
from banbu.turn.model import ProactiveTrigger

from .base import OnHit, SceneRuntime
from .conditions import PreconditionFailed, check_precondition
from .lifecycle import SceneState
from .transitions import matches_step, summarize_change

log = logging.getLogger(__name__)


class WindowedAllSceneRuntime(SceneRuntime):
    def __init__(
        self,
        scene: Scene,
        cache: SnapshotCache,
        *,
        home_id: str,
        on_hit: OnHit | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(scene, on_hit=on_hit)
        if scene.kind != "windowed_all":
            raise ValueError(f"WindowedAllSceneRuntime got kind={scene.kind!r}")
        if not isinstance(scene.trigger, WindowedAllTrigger):
            raise ValueError("windowed_all runtime requires trigger.conditions")
        self._cache = cache
        self._home_id = home_id
        self._clock = clock
        self.state = SceneState()
        self._condition_hits: dict[int, float] = {}
        self._condition_summaries: dict[int, str] = {}
        self._condition_event_ids: dict[int, str] = {}

    def on_event(self, event: DeviceEvent, change: FieldChange) -> None:
        scene = self.scene
        trigger = self._trigger
        st = self.state
        now = self._clock()

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

        matched_any = False
        for idx, condition in enumerate(trigger.conditions):
            if matches_step(condition, event, change):
                matched_any = True
                self._condition_hits[idx] = now
                self._condition_summaries[idx] = summarize_change(event, change)
                self._condition_event_ids[idx] = event.event_id
                log.info(
                    "scene %s: windowed condition %d/%d matched on %s.%s",
                    scene.scene_id,
                    idx + 1,
                    len(trigger.conditions),
                    event.friendly_name,
                    change.field,
                )

        if not matched_any:
            return

        self._drop_expired_hits(now)
        if len(self._condition_hits) != len(trigger.conditions):
            return

        hit_times = list(self._condition_hits.values())
        if max(hit_times) - min(hit_times) > trigger.window_seconds:
            self._drop_expired_hits(now)
            return

        self._evaluate_and_emit()

    @property
    def _trigger(self) -> WindowedAllTrigger:
        assert isinstance(self.scene.trigger, WindowedAllTrigger)
        return self.scene.trigger

    def _drop_expired_hits(self, now: float) -> None:
        window = self._trigger.window_seconds
        for idx, hit_at in list(self._condition_hits.items()):
            if now - hit_at > window:
                self._condition_hits.pop(idx, None)
                self._condition_summaries.pop(idx, None)
                self._condition_event_ids.pop(idx, None)

    def _evaluate_and_emit(self) -> None:
        scene = self.scene
        now = self._clock()

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
            self._clear_hits()
            return

        if not all_pass:
            log.info(
                "scene %s: preconditions rejected - no HIT. results: %s",
                scene.scene_id,
                "; ".join(results),
            )
            self._clear_hits()
            return

        facts = self._collect_facts()
        trigger = ProactiveTrigger(
            scene_id=scene.scene_id,
            home_id=self._home_id,
            facts=facts,
            source_event_summaries=[
                self._condition_summaries[idx]
                for idx in sorted(self._condition_summaries)
            ],
            source_event_ids=[
                self._condition_event_ids[idx]
                for idx in sorted(self._condition_event_ids)
            ],
            triggered_at=now,
        )
        self.state.set_inflight(scene.policy.inflight_seconds, now=now)
        log.info(
            "Scene HIT %s trigger_id=%s inflight_until=%.0f facts=%s",
            scene.scene_id,
            trigger.trigger_id,
            self.state.inflight_until,
            facts,
        )
        self._clear_hits()

        if self._on_hit is not None:
            try:
                self._on_hit(trigger)
            except Exception:
                log.exception("on_hit handler raised (continuing)")

    def _clear_hits(self) -> None:
        self._condition_hits.clear()
        self._condition_summaries.clear()
        self._condition_event_ids.clear()

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
