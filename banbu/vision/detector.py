from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI

from banbu.config.settings import Settings

log = logging.getLogger(__name__)

HAND_ON_CHEEK_SCENE_ID = "hand_on_cheek_color_temp_light_v1"


@dataclass
class VisionDetection:
    scene_id: str | None
    confidence: float
    reason: str


def _build_detection_prompt() -> str:
    return f"""
# Role
You are a home vision detector. Decide whether the current camera frame matches the scene below.

# Scene
id: {HAND_ON_CHEEK_SCENE_ID}
name: person resting their cheek on their hand
criteria:
- Match only when a person is clearly supporting their cheek, chin, or side of face with a hand.
- The hand must be touching and holding up the face, like a resting or thinking pose.
- A hand merely near the face, waving, scratching, drinking, or holding a phone is not a match.
- If unsure, output null.

# Output
Return only JSON:
{{
  "scene_id": "{HAND_ON_CHEEK_SCENE_ID}" or null,
  "confidence": 0.0,
  "reason": "short reason"
}}
""".strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return json.loads(text[start : end + 1])
        raise


def _normalize_detection(raw_text: str) -> VisionDetection:
    data = _parse_json_object(raw_text)
    scene_id = data.get("scene_id") or None
    if scene_id != HAND_ON_CHEEK_SCENE_ID:
        scene_id = None

    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    reason = str(data.get("reason", "")).strip()[:120]
    if scene_id is None:
        confidence = 0.0
        reason = reason or "no matching scene"
    return VisionDetection(scene_id=scene_id, confidence=confidence, reason=reason)


class VisionDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.vision_vlm_base_url,
            api_key=settings.vision_vlm_api_key or "EMPTY",
            timeout=settings.vision_vlm_timeout_seconds,
            http_client=httpx.AsyncClient(
                timeout=settings.vision_vlm_timeout_seconds,
                trust_env=False,
            ),
        )

    async def detect(self, image_data_url: str) -> VisionDetection:
        resp = await self._client.chat.completions.create(
            model=self._settings.vision_vlm_model,
            messages=[
                {"role": "system", "content": _build_detection_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect the configured scene in this frame."},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=300,
        )
        raw_text = resp.choices[0].message.content or ""
        try:
            detection = _normalize_detection(raw_text)
        except Exception:
            log.warning("vision detector returned unparsable content: %r", raw_text)
            detection = VisionDetection(scene_id=None, confidence=0.0, reason="invalid VLM JSON")
        log.info(
            "vision detector result scene_id=%r confidence=%.2f reason=%s",
            detection.scene_id,
            detection.confidence,
            detection.reason,
        )
        return detection
