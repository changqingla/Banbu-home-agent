from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI

from banbu.config.settings import Settings
from banbu.scenes.definition import Scene, VisionTrigger

log = logging.getLogger(__name__)


@dataclass
class VisionDetection:
    scene_id: str | None
    confidence: float
    reason: str


def vision_scenes_for_device(scenes: list[Scene], device_id: str) -> list[Scene]:
    return [
        scene
        for scene in scenes
        if scene.kind == "vision_match"
        and isinstance(scene.trigger, VisionTrigger)
        and scene.trigger.device == device_id
    ]


def _scene_criteria(scene: Scene) -> list[str]:
    criteria = [item.strip() for item in scene.vision_criteria if item.strip()]
    if criteria:
        return criteria
    values = [scene.intent.strip(), scene.name.strip()]
    return [value for value in values if value]


def build_detection_prompt(scenes: list[Scene]) -> str:
    scene_blocks: list[str] = []
    for scene in scenes:
        criteria = "\n".join(f"- {item}" for item in _scene_criteria(scene))
        scene_blocks.append(
            "\n".join(
                [
                    f"id: {scene.scene_id}",
                    f"name: {scene.name}",
                    "criteria:",
                    criteria or "- Match only when this scene is visually clear.",
                ]
            )
        )
    scenes_text = "\n\n".join(scene_blocks)
    scene_ids = ", ".join(scene.scene_id for scene in scenes)

    return f"""
# Role
You are a home vision detector. Decide whether the current camera frame matches exactly one configured scene.

# Scenes
{scenes_text}

# Decision rules
- Return a scene_id only when exactly one configured scene clearly matches the frame.
- If no scene matches, multiple scenes match, or you are unsure, return null.
- Valid scene_id values: {scene_ids}

# Output
Return only JSON:
{{
  "scene_id": "<one configured scene_id>" or null,
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


def normalize_detection(raw_text: str, allowed_scene_ids: set[str]) -> VisionDetection:
    data = _parse_json_object(raw_text)
    raw_scene_id = data.get("scene_id")
    scene_id = str(raw_scene_id).strip() if raw_scene_id else None
    if scene_id not in allowed_scene_ids:
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
    def __init__(self, settings: Settings, scenes: list[Scene]) -> None:
        if not scenes:
            raise ValueError(f"no vision_match scenes configured for device {settings.vision_device_id!r}")
        self._settings = settings
        self._scenes = scenes
        self._allowed_scene_ids = {scene.scene_id for scene in scenes}
        self._prompt = build_detection_prompt(scenes)
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
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect the configured scene, or return null."},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=300,
        )
        raw_text = resp.choices[0].message.content or ""
        try:
            detection = normalize_detection(raw_text, self._allowed_scene_ids)
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
