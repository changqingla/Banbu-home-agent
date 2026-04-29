from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from banbu.config.settings import Settings
from banbu.scenes.definition import Scene

from .detector import VisionDetector, vision_scenes_for_device
from .publisher import BatchEventPublisher
from .rtsp_monitor import run_rtsp_monitor

log = logging.getLogger(__name__)


class VisionService:
    def __init__(self, settings: Settings, scenes: list[Scene]) -> None:
        self._settings = settings
        self._scenes = vision_scenes_for_device(scenes, settings.vision_device_id)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if not self._settings.vision_enabled:
            log.info("vision service disabled")
            return
        if not self._scenes:
            raise RuntimeError(
                f"vision service enabled but no vision_match scenes reference "
                f"{self._settings.vision_device_id!r}"
            )
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())
        log.info("vision service started")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        detector = VisionDetector(self._settings, self._scenes)
        publisher = BatchEventPublisher(self._settings)
        frame_count = 0

        async def process_frame(data_url: str) -> None:
            nonlocal frame_count
            frame_count += 1
            log.info(
                "vision: sending frame #%d to VLM model=%s image_bytes_base64=%d",
                frame_count,
                self._settings.vision_vlm_model,
                len(data_url),
            )
            detection = await detector.detect(data_url)
            await publisher.publish(detection)

        await run_rtsp_monitor(self._settings, process_frame)
