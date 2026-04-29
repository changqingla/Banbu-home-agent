import pytest

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.reactive.scene_matcher import SceneMatchError, rank_scene_candidates, select_scene_match
from banbu.scenes.definition import Scene


def _device(
    friendly_name: str,
    *,
    local_id: int,
    room: str,
    aliases: list[str],
) -> ResolvedDevice:
    return ResolvedDevice(
        spec=DeviceSpec(
            friendly_name=friendly_name,
            room=room,
            role="light_switch",
            aliases=aliases,
        ),
        local_id=local_id,
        ieee_address=f"0x{local_id}",
        model="test",
        capabilities={"state"},
    )


def _scene(
    scene_id: str = "entry_auto_light_v1",
    *,
    name: str = "进门自动开灯",
    intent: str = "有人进门且光线较暗时打开玄关灯",
    device: str = "switch_entry_light",
) -> Scene:
    return Scene.model_validate(
        {
            "scene_id": scene_id,
            "name": name,
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": device,
                        "field": "payload.state",
                        "transition": "OFF->ON",
                    }
                ]
            },
            "context_devices": {"trigger": [device], "context_only": []},
            "intent": intent,
            "actions_hint": [
                {
                    "tool": "execute_plan",
                    "args": {"device": device, "action": "turn_on"},
                }
            ],
        }
    )


def test_matches_scene_by_id_and_intent_like_text() -> None:
    resolver = DeviceResolver(
        [_device("switch_entry_light", local_id=2, room="玄关", aliases=["玄关灯"])]
    )
    scene = _scene()

    by_id = rank_scene_candidates("entry_auto_light_v1", [scene], resolver)
    by_intent = rank_scene_candidates("进门开灯", [scene], resolver)

    assert by_id[0].scene.scene_id == "entry_auto_light_v1"
    assert by_id[0].reasons[0].startswith("scene_id:direct")
    assert by_intent[0].scene.scene_id == "entry_auto_light_v1"
    assert any(reason.startswith(("name:overlap", "intent:overlap")) for reason in by_intent[0].reasons)


def test_matches_scene_by_device_alias_room_and_role() -> None:
    resolver = DeviceResolver(
        [_device("switch_entry_light", local_id=2, room="玄关", aliases=["玄关灯"])]
    )
    scene = _scene(name="Entry automation", intent="")

    candidate = select_scene_match("打开玄关灯", [scene], resolver)

    assert candidate.scene.scene_id == "entry_auto_light_v1"
    assert any(
        reason.startswith(("device:alias", "device:room+role"))
        for reason in candidate.reasons
    )


def test_ambiguous_scene_match_fails_closed() -> None:
    resolver = DeviceResolver(
        [_device("switch_entry_light", local_id=2, room="玄关", aliases=["玄关灯"])]
    )
    scenes = [
        _scene("entry_auto_light_v1", name="Entry A", intent=""),
        _scene("entry_evening_light_v1", name="Entry B", intent=""),
    ]

    with pytest.raises(SceneMatchError) as exc:
        select_scene_match("打开玄关灯", scenes, resolver)

    assert exc.value.kind == "ambiguous_scene"
    assert exc.value.candidates == ("entry_auto_light_v1", "entry_evening_light_v1")
