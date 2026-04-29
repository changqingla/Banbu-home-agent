from __future__ import annotations

from .model import ProactiveTrigger, Turn


def from_trigger(trigger: ProactiveTrigger) -> Turn:
    return Turn.from_proactive(trigger)


def from_reactive(
    utterance: str,
    *,
    home_id: str,
    user_id: str,
    source: str = "cli",
) -> Turn:
    return Turn.from_reactive(
        utterance,
        home_id=home_id,
        user_id=user_id,
        source=source,
    )
