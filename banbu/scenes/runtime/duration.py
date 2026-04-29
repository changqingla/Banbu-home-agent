from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import DurationTrigger, Scene
from banbu.state.snapshot_cache import MISSING, SnapshotCache
from banbu.turn.model import ProactiveTrigger

from .base import OnHit, SceneRuntime
from .conditions import PreconditionFailed, check_precondition
from .lifecycle import SceneState

log = logging.getLogger(__name__)


class DurationSceneRuntime(SceneRuntime):
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
        if scene.kind != "duration_triggered":
            raise ValueError(f"DurationSceneRuntime got kind={scene.kind!r}")
        if not isinstance(scene.trigger, DurationTrigger):
            raise ValueError("duration runtime requires trigger.condition")
        self._cache = cache
        self._home_id = home_id
        self._clock = clock
        self.state = SceneState()
        self.condition_satisfied_since: float | None = None

    def on_event(self, event: DeviceEvent, change: FieldChange) -> None:
        trigger = self._trigger
        if event.friendly_name != trigger.condition.device:
            return
        leaf = self._leaf_field(trigger.condition.field)
        if change.field != leaf:
            return
        self._evaluate_condition(source=f"event {event.friendly_name}.{change.field}")

    def on_tick(self) -> None:
        self._evaluate_condition(source="tick")

    @property
    def _trigger(self) -> DurationTrigger:
        assert isinstance(self.scene.trigger, DurationTrigger)
        return self.scene.trigger

    def _evaluate_condition(self, *, source: str) -> None:
        scene = self.scene
        st = self.state
        now = self._clock()
        satisfied, current_value = self._condition_satisfied()

        if not satisfied:
            if self.condition_satisfied_since is not None:
                log.info(
                    "scene %s: duration condition reset by %s (value=%r)",
                    scene.scene_id,
                    source,
                    current_value,
                )
            self.condition_satisfied_since = None
            return

        if self.condition_satisfied_since is None:
            self.condition_satisfied_since = now
            log.info(
                "scene %s: duration condition became true at %.0f by %s",
                scene.scene_id,
                now,
                source,
            )

        elapsed = now - self.condition_satisfied_since
        if elapsed < self._trigger.duration_seconds:
            return

        if st.is_in_cooldown(now):
            log.info(
                "scene %s: cooldown active (%.1fs left), duration hit suppressed",
                scene.scene_id,
                st.cooldown_until - now,
            )
            return
        if st.is_inflight(now):
            log.info(
                "scene %s: inflight active (%.1fs left), duration hit suppressed",
                scene.scene_id,
                st.inflight_until - now,
            )
            return

        self._evaluate_and_emit(now=now, elapsed=elapsed)

    def _condition_satisfied(self) -> tuple[bool, Any]:
        condition = self._trigger.condition
        value, _ = self._cache.field(condition.device, condition.field)
        if value is MISSING:
            return False, MISSING
        return value == condition.value, value

    def _evaluate_and_emit(self, *, now: float, elapsed: float) -> None:
        scene = self.scene

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
            self.condition_satisfied_since = None
            return

        if not all_pass:
            log.info(
                "scene %s: preconditions rejected - no HIT. results: %s",
                scene.scene_id,
                "; ".join(results),
            )
            self.condition_satisfied_since = None
            return

        facts = self._collect_facts(elapsed=elapsed)
        trigger = ProactiveTrigger(
            scene_id=scene.scene_id,
            home_id=self._home_id,
            facts=facts,
            source_event_summaries=[
                (
                    f"{self._trigger.condition.device}.{self._leaf_field(self._trigger.condition.field)} "
                    f"held {self._trigger.condition.value!r} for {elapsed:.1f}s"
                )
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
        self.condition_satisfied_since = None

        if self._on_hit is not None:
            try:
                self._on_hit(trigger)
            except Exception:
                log.exception("on_hit handler raised (continuing)")

    def _collect_facts(self, *, elapsed: float) -> dict[str, Any]:
        facts: dict[str, Any] = {
            "duration": {
                "condition_satisfied_since": self.condition_satisfied_since,
                "elapsed_seconds": elapsed,
                "required_seconds": self._trigger.duration_seconds,
            }
        }
        for name in self.scene.trigger_devices():
            snap = self._cache.get_by_name(name)
            if snap is not None:
                facts[name] = dict(snap.payload)
        for pre in self.scene.preconditions:
            value, _ = self._cache.field(pre.device, pre.field)
            facts.setdefault(pre.device, {})[pre.field] = None if value is MISSING else value
        return facts

    def _leaf_field(self, field: str) -> str:
        return field[len("payload."):] if field.startswith("payload.") else field
