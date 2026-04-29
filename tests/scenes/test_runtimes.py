from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene
from banbu.scenes.runtime.sequential import SequentialSceneRuntime
from banbu.scenes.runtime.vision_match import VisionMatchSceneRuntime
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import ProactiveTrigger


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="door_sensor", role="door_sensor"),
                local_id=1,
                ieee_address="0x1",
                model="test",
                capabilities={"contact"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="entry_light", role="light_switch"),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="entry_camera_vision_1", role="vision_detector"),
                local_id=-1001,
                ieee_address="virtual:entry_camera_vision_1",
                model="virtual",
                capabilities={"scene_id", "detected", "confidence", "frame_id"},
            ),
        ]
    )


def test_sequential_runtime_emits_trigger_after_matching_steps_and_preconditions() -> None:
    cache = SnapshotCache(_resolver())
    cache.update(1, {"contact": True}, source="test")
    cache.update(2, {"state": "OFF"}, source="test")
    scene = Scene.model_validate(
        {
            "scene_id": "entry_scene",
            "name": "Entry scene",
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": "door_sensor",
                        "field": "payload.contact",
                        "transition": "false->true",
                    }
                ]
            },
            "context_devices": {"trigger": ["door_sensor"], "context_only": ["entry_light"]},
            "preconditions": [
                {
                    "device": "entry_light",
                    "field": "payload.state",
                    "op": "neq",
                    "value": "ON",
                }
            ],
            "policy": {"inflight_seconds": 20},
        }
    )
    hits: list[ProactiveTrigger] = []
    runtime = SequentialSceneRuntime(scene, cache, home_id="home_a", on_hit=hits.append)

    runtime.on_event(
        DeviceEvent(
            local_id=1,
            friendly_name="door_sensor",
            ieee_address="0x1",
            payload={"contact": True},
            changes=[],
        ),
        FieldChange(field="contact", old=False, new=True),
    )

    assert len(hits) == 1
    assert hits[0].scene_id == "entry_scene"
    assert hits[0].facts["door_sensor"] == {"contact": True}
    assert runtime.state.is_inflight()


def test_vision_match_runtime_requires_consecutive_hits() -> None:
    cache = SnapshotCache(_resolver())
    scene = Scene.model_validate(
        {
            "scene_id": "hand_pose",
            "name": "Hand pose",
            "kind": "vision_match",
            "trigger": {
                "device": "entry_camera_vision_1",
                "field": "payload.scene_id",
                "value": "hand_pose",
                "confidence_field": "payload.confidence",
                "detected_field": "payload.detected",
                "frame_id_field": "payload.frame_id",
            },
            "vision_policy": {"confidence_threshold": 0.7, "consecutive_hits": 2},
        }
    )
    hits: list[ProactiveTrigger] = []
    runtime = VisionMatchSceneRuntime(scene, cache, home_id="home_a", on_hit=hits.append)

    first = DeviceEvent(
        local_id=-1001,
        friendly_name="entry_camera_vision_1",
        ieee_address="virtual:entry_camera_vision_1",
        payload={"scene_id": "hand_pose", "detected": True, "confidence": 0.8, "frame_id": "f1"},
        changes=[],
        sequence=1,
        source="vision",
    )
    second = DeviceEvent(
        local_id=-1001,
        friendly_name="entry_camera_vision_1",
        ieee_address="virtual:entry_camera_vision_1",
        payload={"scene_id": "hand_pose", "detected": True, "confidence": 0.9, "frame_id": "f2"},
        changes=[],
        sequence=2,
        source="vision",
    )

    runtime.on_event(first, FieldChange(field="scene_id", old=None, new="hand_pose"))
    runtime.on_event(second, FieldChange(field="scene_id", old=None, new="hand_pose"))

    assert len(hits) == 1
    assert hits[0].facts["vision"]["scene_id"] == "hand_pose"
    assert hits[0].facts["vision"]["confidence"] == 0.9
