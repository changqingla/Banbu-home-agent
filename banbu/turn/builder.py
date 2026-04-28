from __future__ import annotations

from .model import ProactiveTrigger, Turn


def from_trigger(trigger: ProactiveTrigger) -> Turn:
    return Turn.from_proactive(trigger)
