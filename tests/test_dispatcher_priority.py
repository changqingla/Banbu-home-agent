from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.dispatcher import Dispatcher
from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene
from banbu.scenes.reverse_index import build_reverse_index
from banbu.state.snapshot_cache import SnapshotCache


class FakeRuntime:
    def __init__(self, scene_id: str, calls: list[str]) -> None:
        self.scene_id = scene_id
        self.calls = calls

    def on_event(self, event, change) -> None:
        self.calls.append(self.scene_id)


def _scene(scene_id: str, priority: int) -> Scene:
    return Scene.model_validate(
        {
            "scene_id": scene_id,
            "name": scene_id,
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": "entry_sensor",
                        "field": "payload.contact",
                        "transition": "false->true",
                    }
                ]
            },
            "policy": {"priority": priority},
        }
    )


def test_dispatcher_routes_trigger_scenes_by_priority() -> None:
    scenes = [_scene("low_scene", 1), _scene("high_scene", 10)]
    resolver = DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="entry_sensor",
                    role="door_sensor",
                    care_fields=["contact"],
                ),
                local_id=1,
                ieee_address="0x1",
                model="test",
                capabilities={"contact"},
            )
        ]
    )
    dispatcher = Dispatcher(
        scenes,
        build_reverse_index(scenes, resolver),
        SnapshotCache(resolver),
        home_id="home_a",
    )
    calls: list[str] = []
    dispatcher._runtimes = {  # noqa: SLF001
        "low_scene": FakeRuntime("low_scene", calls),
        "high_scene": FakeRuntime("high_scene", calls),
    }
    event = DeviceEvent(
        local_id=1,
        friendly_name="entry_sensor",
        ieee_address="0x1",
        payload={"contact": True},
        changes=[FieldChange(field="contact", old=False, new=True)],
    )

    dispatcher.on_event(event)

    assert calls == ["high_scene", "low_scene"]
