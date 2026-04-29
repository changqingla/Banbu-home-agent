from banbu.scenes.definition import Scene
from banbu.vision.detector import build_detection_prompt, normalize_detection, vision_scenes_for_device


def _vision_scene(
    scene_id: str,
    *,
    device: str = "entry_camera_vision_1",
    criteria: list[str] | None = None,
) -> Scene:
    return Scene.model_validate(
        {
            "scene_id": scene_id,
            "name": f"Vision scene {scene_id}",
            "kind": "vision_match",
            "trigger": {
                "device": device,
                "field": "payload.scene_id",
                "value": scene_id,
            },
            "vision_criteria": criteria or [f"Detect {scene_id} only when visually clear."],
            "context_devices": {"trigger": [device], "context_only": []},
            "intent": f"Detect {scene_id}",
        }
    )


def test_vision_scenes_for_device_filters_configured_camera() -> None:
    entry_scene = _vision_scene("entry_scene", device="entry_camera_vision_1")
    kitchen_scene = _vision_scene("kitchen_scene", device="kitchen_camera_vision_1")

    scenes = vision_scenes_for_device([entry_scene, kitchen_scene], "entry_camera_vision_1")

    assert [scene.scene_id for scene in scenes] == ["entry_scene"]


def test_build_detection_prompt_lists_all_configured_scenes_and_criteria() -> None:
    scenes = [
        _vision_scene("hand_on_cheek", criteria=["Person supports cheek with hand."]),
        _vision_scene("standing_near_door", criteria=["Person is standing near the entry door."]),
    ]

    prompt = build_detection_prompt(scenes)

    assert "hand_on_cheek" in prompt
    assert "standing_near_door" in prompt
    assert "Person supports cheek with hand." in prompt
    assert "Person is standing near the entry door." in prompt
    assert "Return a scene_id only when exactly one configured scene clearly matches" in prompt


def test_normalize_detection_accepts_allowed_scene_and_clamps_confidence() -> None:
    detection = normalize_detection(
        '{"scene_id": "standing_near_door", "confidence": 1.7, "reason": "clear"}',
        {"hand_on_cheek", "standing_near_door"},
    )

    assert detection.scene_id == "standing_near_door"
    assert detection.confidence == 1.0
    assert detection.reason == "clear"


def test_normalize_detection_rejects_unconfigured_scene() -> None:
    detection = normalize_detection(
        '{"scene_id": "unknown_scene", "confidence": 0.9, "reason": "not configured"}',
        {"hand_on_cheek"},
    )

    assert detection.scene_id is None
    assert detection.confidence == 0.0
    assert detection.reason == "not configured"


def test_normalize_detection_extracts_json_object_from_chatty_response() -> None:
    detection = normalize_detection(
        'Here is the result: {"scene_id": null, "confidence": 0.4, "reason": "uncertain"}',
        {"hand_on_cheek"},
    )

    assert detection.scene_id is None
    assert detection.confidence == 0.0
    assert detection.reason == "uncertain"
