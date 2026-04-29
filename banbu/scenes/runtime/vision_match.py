from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene, VisionTrigger
from banbu.state.snapshot_cache import MISSING, SnapshotCache
from banbu.turn.model import ProactiveTrigger

from .base import OnHit, SceneRuntime
from .conditions import PreconditionFailed, check_precondition
from .lifecycle import SceneState

log = logging.getLogger(__name__)


def _dig(payload: dict[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    if parts and parts[0] == "payload":
        parts = parts[1:]
    cur: Any = payload
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def _change_key(event: DeviceEvent, trigger: VisionTrigger) -> str:
    frame_id = _dig(event.payload, trigger.frame_id_field)
    if frame_id is not MISSING and frame_id is not None:
        return f"frame:{frame_id}"
    if event.sequence is not None:
        return f"seq:{event.sequence}"
    return f"ts:{event.timestamp:.6f}"


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class VisionMatchSceneRuntime(SceneRuntime):
    def __init__(
        self,
        scene: Scene,
        cache: SnapshotCache,
        *,
        home_id: str,
        on_hit: OnHit | None = None,
    ) -> None:
        super().__init__(scene, on_hit=on_hit)
        if scene.kind != "vision_match":
            raise ValueError(f"VisionMatchSceneRuntime got kind={scene.kind!r}")
        if not isinstance(scene.trigger, VisionTrigger):
            raise ValueError("vision_match runtime requires trigger.device/field/value")
        self._cache = cache
        self._home_id = home_id
        self.state = SceneState()
        self.positive_streak = 0
        self.last_seen_scene_id: str | None = None
        self.last_detection_at: float | None = None
        self._last_processed_key: str | None = None

    def on_event(self, event: DeviceEvent, change: FieldChange) -> None:
        trigger = self._trigger
        if event.friendly_name != trigger.device:
            return

        key = _change_key(event, trigger)
        if key == self._last_processed_key:
            return
        self._last_processed_key = key

        now = time.time()
        st = self.state
        scene = self.scene

        if st.is_in_cooldown(now):
            log.info(
                "scene %s: cooldown active (%.1fs left), dropping vision event %s",
                scene.scene_id, st.cooldown_until - now, key,
            )
            return
        if st.is_inflight(now):
            log.info(
                "scene %s: inflight active (%.1fs left), dropping vision event %s",
                scene.scene_id, st.inflight_until - now, key,
            )
            return

        scene_id = _dig(event.payload, trigger.field)
        detected = _dig(event.payload, trigger.detected_field)
        confidence = _dig(event.payload, trigger.confidence_field)
        try:
            confidence_value = float(0 if confidence is MISSING or confidence is None else confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0

        matched = (
            scene_id == trigger.value
            and detected is True
            and confidence_value >= scene.vision_policy.confidence_threshold
        )
        self.last_seen_scene_id = None if scene_id is MISSING else str(scene_id)
        self.last_detection_at = now

        if not matched:
            if scene.vision_policy.reset_on_miss:
                self.positive_streak = 0
            log.info(
                "scene %s: vision miss scene_id=%r detected=%r confidence=%.2f streak=%d",
                scene.scene_id, None if scene_id is MISSING else scene_id,
                detected, confidence_value, self.positive_streak,
            )
            return

        self.positive_streak += 1
        log.info(
            "scene %s: vision matched %s confidence=%.2f streak=%d/%d",
            scene.scene_id, trigger.value, confidence_value,
            self.positive_streak, scene.vision_policy.consecutive_hits,
        )

        if self.positive_streak < scene.vision_policy.consecutive_hits:
            return

        self._evaluate_and_emit(event)

    @property
    def _trigger(self) -> VisionTrigger:
        assert isinstance(self.scene.trigger, VisionTrigger)
        return self.scene.trigger

    def _evaluate_and_emit(self, event: DeviceEvent) -> None:
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
            self.positive_streak = 0
            return

        if not all_pass:
            log.info(
                "scene %s: preconditions rejected - no HIT. results: %s",
                scene.scene_id, "; ".join(results),
            )
            self.positive_streak = 0
            return

        triggered_at = time.time()
        facts = self._collect_facts(event, triggered_at=triggered_at)
        trigger = ProactiveTrigger(
            scene_id=scene.scene_id,
            home_id=self._home_id,
            facts=facts,
            source_event_summaries=[self._summary(facts)],
            source_event_ids=[event.event_id],
            triggered_at=triggered_at,
        )
        st.set_inflight(scene.policy.inflight_seconds)
        self.positive_streak = 0
        log.info(
            "Scene HIT %s trigger_id=%s triggered_at=%s inflight_until=%.0f facts=%s",
            scene.scene_id, trigger.trigger_id, _iso_from_ts(triggered_at), st.inflight_until, facts,
        )

        if self._on_hit is not None:
            try:
                self._on_hit(trigger)
            except Exception:
                log.exception("on_hit handler raised (continuing)")

    def _collect_facts(self, event: DeviceEvent, *, triggered_at: float | None = None) -> dict[str, Any]:
        trigger = self._trigger
        payload = event.payload
        vision: dict[str, Any] = {
            "camera": event.friendly_name,
            "scene_id": _dig(payload, trigger.field),
            "detected": _dig(payload, trigger.detected_field),
            "confidence": _dig(payload, trigger.confidence_field),
            "reason": payload.get("reason"),
            "frame_id": _dig(payload, trigger.frame_id_field),
            "frame_at": payload.get("frame_at"),
            "sequence": event.sequence,
            "source": event.source,
        }
        if triggered_at is not None:
            vision["scene_triggered_at"] = triggered_at
            vision["scene_triggered_at_iso"] = _iso_from_ts(triggered_at)
        return {"vision": vision}

    def _summary(self, facts: dict[str, Any]) -> str:
        vision = facts["vision"]
        return (
            f"{vision['camera']} vision scene={vision['scene_id']!r} "
            f"confidence={vision['confidence']!r} frame={vision['frame_id']!r}"
        )
