from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import WILDCARD, Scene, parse_value
from banbu.scenes.reverse_index import build_reverse_index


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="door_sensor", role="door_sensor", care_fields=["contact"]),
                local_id=1,
                ieee_address="0x1",
                model="test",
                capabilities={"contact"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="entry_light", role="light_switch", care_fields=["state"]),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state"},
            ),
        ]
    )


def test_parse_value_preserves_on_off_strings() -> None:
    assert parse_value("true") is True
    assert parse_value("42") == 42
    assert parse_value("*") is WILDCARD
    assert parse_value("ON") == "ON"
    assert parse_value("OFF") == "OFF"


def test_reverse_index_marks_trigger_and_context_only_devices() -> None:
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
            "context_devices": {"context_only": ["entry_light"]},
        }
    )

    index = build_reverse_index([scene], _resolver())

    assert index.lookup("door_sensor", "payload.contact") == [("entry_scene", "trigger")]
    assert index.lookup("entry_light", "payload.state") == [("entry_scene", "context_only")]
