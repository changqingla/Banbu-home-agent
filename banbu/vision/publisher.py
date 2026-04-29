from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from banbu.config.settings import Settings

from .detector import VisionDetection


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class BatchEventPublisher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sequence = 0
        self._previous_values: dict[str, Any] = {
            "scene_id": None,
            "detected": False,
            "confidence": 0,
            "reason": "no matching scene",
            "frame_id": None,
            "frame_at": 0,
        }

    async def publish(self, detection: VisionDetection) -> None:
        self._sequence += 1
        frame_at = time.time()
        frame_id = f"vision_{self._sequence}_{int(frame_at * 1000)}"
        values = {
            "scene_id": detection.scene_id,
            "detected": detection.scene_id is not None,
            "confidence": detection.confidence,
            "reason": detection.reason,
            "frame_id": frame_id,
            "frame_at": frame_at,
        }
        body = {
            "changed_at": _iso_now(),
            "reported_at": _iso_now(),
            "source": "vision",
            "payload": [
                {
                    "device_id": self._settings.vision_device_id,
                    "sequence": self._sequence,
                    "values": values,
                    "previous_values": self._previous_values,
                }
            ],
        }
        url = f"{self._settings.vision_post_base_url.rstrip('/')}{self._settings.webhook_path}"
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
        self._previous_values = values

