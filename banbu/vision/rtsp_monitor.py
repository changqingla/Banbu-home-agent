from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import suppress
from typing import Awaitable, Callable

from banbu.config.settings import Settings

log = logging.getLogger(__name__)

FrameProcessor = Callable[[str], Awaitable[None]]


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Vision monitoring requires opencv-python-headless to be installed.") from exc
    return cv2


async def _open_capture(settings: Settings):
    cv2 = _import_cv2()
    capture = await asyncio.to_thread(cv2.VideoCapture, settings.vision_rtsp_url)
    try:
        await asyncio.to_thread(capture.set, cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return capture


async def _read_frame(capture):
    return await asyncio.to_thread(capture.read)


async def _release_capture(capture) -> None:
    await asyncio.to_thread(capture.release)


async def _encode_frame_to_data_url(frame, settings: Settings) -> str | None:
    cv2 = _import_cv2()
    ok, encoded = await asyncio.to_thread(
        cv2.imencode,
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), settings.vision_jpeg_quality],
    )
    if not ok:
        return None
    payload = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def _detect_motion(prev_frame, curr_frame, settings: Settings) -> bool:
    cv2 = _import_cv2()
    size = (settings.vision_motion_sample_size, settings.vision_motion_sample_size)
    prev_gray = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), size)
    curr_gray = cv2.resize(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), size)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(
        diff,
        settings.vision_motion_pixel_diff_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    changed_pixels = cv2.countNonZero(thresh)
    return changed_pixels / (size[0] * size[1]) >= settings.vision_motion_changed_ratio_threshold


async def _run_processor_safely(processor: FrameProcessor, data_url: str) -> None:
    try:
        await processor(data_url)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("vision frame processor failed")


async def run_rtsp_monitor(settings: Settings, processor: FrameProcessor) -> None:
    if not settings.vision_rtsp_url:
        log.info("vision monitor skipped: BANBU_VISION_RTSP_URL is empty")
        return

    processor_task: asyncio.Task | None = None
    while True:
        capture = None
        prev_frame = None
        try:
            capture = await _open_capture(settings)
            if not capture or not capture.isOpened():
                log.warning("failed to open vision RTSP stream, retrying later")
                if capture:
                    await _release_capture(capture)
                await asyncio.sleep(settings.vision_reconnect_seconds)
                continue

            log.info("vision RTSP stream connected")
            last_preview_at = 0.0
            loop = asyncio.get_running_loop()
            while True:
                ok, frame = await _read_frame(capture)
                if not ok or frame is None:
                    log.warning("failed to read vision RTSP frame, reconnecting")
                    break

                now = loop.time()
                if now - last_preview_at < settings.vision_frame_interval_seconds:
                    continue

                should_process = prev_frame is None or _detect_motion(prev_frame, frame, settings)
                prev_frame = frame
                last_preview_at = now

                if not should_process:
                    continue
                if processor_task is not None and not processor_task.done():
                    log.info("vision motion detected but VLM processor is busy; skipping frame")
                    continue

                data_url = await _encode_frame_to_data_url(frame, settings)
                if data_url is None:
                    continue
                log.info("vision motion detected; dispatching frame to VLM")
                processor_task = asyncio.create_task(_run_processor_safely(processor, data_url))
        except asyncio.CancelledError:
            if processor_task is not None and not processor_task.done():
                processor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await processor_task
            raise
        except Exception:
            log.exception("vision RTSP monitor loop failed, retrying")
            await asyncio.sleep(settings.vision_reconnect_seconds)
        finally:
            if capture is not None:
                with suppress(Exception):
                    await _release_capture(capture)

