from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ProactiveTrigger:
    scene_id: str
    home_id: str
    facts: dict[str, Any]
    source_event_summaries: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    triggered_at: float = field(default_factory=time.time)
    trigger_id: str = field(default_factory=lambda: f"trg_{uuid.uuid4().hex[:12]}")


ThreadType = Literal["proactive", "reactive"]


@dataclass
class Turn:
    turn_id: str
    thread_type: ThreadType
    conversation_id: str
    home_id: str
    user_id: str | None = None
    source: str = "system"
    scene_id: str | None = None
    utterance: str | None = None
    trigger: ProactiveTrigger | None = None

    @property
    def input(self) -> "str | dict[str, Any]":
        if self.thread_type == "reactive":
            return self.utterance or ""
        trg = self.trigger
        if trg is None:
            return {}
        return {
            "trigger_id": trg.trigger_id,
            "scene_id": trg.scene_id,
            "facts": trg.facts,
            "source_event_ids": trg.source_event_ids,
        }

    @classmethod
    def from_proactive(cls, trigger: ProactiveTrigger) -> "Turn":
        return cls(
            turn_id=f"turn_{uuid.uuid4().hex[:12]}",
            thread_type="proactive",
            conversation_id=f"{trigger.home_id}_{trigger.scene_id}",
            home_id=trigger.home_id,
            source="scene",
            scene_id=trigger.scene_id,
            trigger=trigger,
        )

    @classmethod
    def from_reactive(
        cls,
        utterance: str,
        *,
        home_id: str,
        user_id: str,
        source: str = "cli",
    ) -> "Turn":
        utterance = utterance.strip()
        if not utterance:
            raise ValueError("reactive utterance must not be empty")
        return cls(
            turn_id=f"turn_{uuid.uuid4().hex[:12]}",
            thread_type="reactive",
            conversation_id=f"{home_id}_{user_id}",
            home_id=home_id,
            user_id=user_id,
            source=source,
            utterance=utterance,
        )
