from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene
from banbu.turn.model import ProactiveTrigger

OnHit = Callable[[ProactiveTrigger], None]


class SceneRuntime(ABC):
    def __init__(self, scene: Scene, on_hit: OnHit | None = None) -> None:
        self.scene = scene
        self._on_hit = on_hit

    @abstractmethod
    def on_event(self, event: DeviceEvent, change: FieldChange) -> None: ...
