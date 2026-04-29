from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


FeedbackOutcome = Literal["success", "failure", "skipped", "agent_error"]


@dataclass(frozen=True)
class FeedbackEntry:
    home_id: str
    scene_id: str
    trigger_id: str | None
    outcome: FeedbackOutcome
    summary: str
    created_at: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    def __init__(self, *, max_entries_per_scene: int = 5) -> None:
        self._max_entries = max_entries_per_scene
        self._entries: dict[tuple[str, str], deque[FeedbackEntry]] = defaultdict(
            lambda: deque(maxlen=self._max_entries)
        )

    def add(self, entry: FeedbackEntry) -> None:
        self._entries[(entry.home_id, entry.scene_id)].append(entry)

    def recent(self, home_id: str, scene_id: str) -> list[FeedbackEntry]:
        return list(self._entries.get((home_id, scene_id), ()))
