"""Per-scene lifecycle state.

Two timestamps tracked separately (see plan §4.2.1):
  inflight_until — short concurrency lock, written on HIT, cleared on
                   success / failure / skip. Prevents the same event burst
                   from re-triggering the scene mid-execution.
  cooldown_until — business cooldown, written ONLY when execute_plan
                   actually succeeded. Phase 3 has no execution yet, so
                   the cooldown stays unset.

A failure or skip clears inflight without writing cooldown — the next
identical event burst can therefore retry, leaving recovery paths open.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SceneState:
    cursor: int = 0
    last_step_at: float | None = None
    inflight_until: float = 0.0
    cooldown_until: float = 0.0

    def is_in_cooldown(self, now: float | None = None) -> bool:
        return (now or time.time()) < self.cooldown_until

    def is_inflight(self, now: float | None = None) -> bool:
        return (now or time.time()) < self.inflight_until

    def reset_cursor(self) -> None:
        self.cursor = 0
        self.last_step_at = None

    def set_inflight(self, seconds: float, now: float | None = None) -> None:
        self.inflight_until = (now or time.time()) + seconds

    def clear_inflight(self) -> None:
        self.inflight_until = 0.0

    def set_cooldown(self, seconds: float, now: float | None = None) -> None:
        self.cooldown_until = (now or time.time()) + seconds
