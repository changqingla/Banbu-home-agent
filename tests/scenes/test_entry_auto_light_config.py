from pathlib import Path

import yaml

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene
from banbu.scenes.runtime.sequential import SequentialSceneRuntime
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import ProactiveTrigger


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_SCENE_PATH = REPO_ROOT / "banbu/config/scenes/entry_auto_light_v1.yaml"

DOOR = "0xa4c138c1dfcc3e7e"
PIR = "0xa4c1384addf12b0c"
RADAR = "0xa4c138d2ceca243c"
LIGHT = "0xa4c1388ce42078b7"


def _load_entry_scene() -> Scene:
    raw = yaml.safe_load(ENTRY_SCENE_PATH.read_text(encoding="utf-8"))
    return Scene.model_validate(raw)


def _resolver() -> DeviceResolver:
    devices = [
        ResolvedDevice(
            spec=DeviceSpec(friendly_name=DOOR, role="door_sensor", care_fields=["contact"]),
            local_id=4,
            ieee_address=DOOR,
            model="TS0203",
            capabilities={"contact"},
        ),
        ResolvedDevice(
            spec=DeviceSpec(friendly_name=PIR, role="motion_sensor", care_fields=["occupancy"]),
            local_id=1,
            ieee_address=PIR,
            model="IH012-RT02",
            capabilities={"occupancy"},
        ),
        ResolvedDevice(
            spec=DeviceSpec(friendly_name=RADAR, role="presence_radar", care_fields=["presence", "illuminance"]),
            local_id=3,
            ieee_address=RADAR,
            model="TS0601-PIR-Sensor",
            capabilities={"presence", "illuminance"},
        ),
        ResolvedDevice(
            spec=DeviceSpec(
                friendly_name=LIGHT,
                role="color_temp_light",
                care_fields=["state", "brightness", "color_temp"],
                actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
            ),
            local_id=12,
            ieee_address=LIGHT,
            model="TS0502B",
            capabilities={"state", "brightness", "color_temp"},
        ),
    ]
    return DeviceResolver(devices)


def _cache(*, illuminance: int, light_state: str) -> SnapshotCache:
    cache = SnapshotCache(_resolver())
    cache.update(4, {"contact": False}, source="test")
    cache.update(1, {"occupancy": True}, source="test")
    cache.update(3, {"presence": True, "illuminance": illuminance}, source="test")
    cache.update(12, {"state": light_state, "brightness": 120, "color_temp": 370}, source="test")
    return cache


def _door_open_event() -> tuple[DeviceEvent, FieldChange]:
    change = FieldChange(field="contact", old=True, new=False)
    event = DeviceEvent(
        local_id=4,
        friendly_name=DOOR,
        ieee_address=DOOR,
        payload={"contact": False},
        changes=[change],
    )
    return event, change


def _pir_occupied_event() -> tuple[DeviceEvent, FieldChange]:
    change = FieldChange(field="occupancy", old=False, new=True)
    event = DeviceEvent(
        local_id=1,
        friendly_name=PIR,
        ieee_address=PIR,
        payload={"occupancy": True},
        changes=[change],
    )
    return event, change


def _run_entry_sequence(*, illuminance: int, light_state: str) -> list[ProactiveTrigger]:
    hits: list[ProactiveTrigger] = []
    runtime = SequentialSceneRuntime(
        _load_entry_scene(),
        _cache(illuminance=illuminance, light_state=light_state),
        home_id="home_default",
        on_hit=hits.append,
    )
    door_event, door_change = _door_open_event()
    pir_event, pir_change = _pir_occupied_event()

    runtime.on_event(door_event, door_change)
    runtime.on_event(pir_event, pir_change)

    return hits


def test_entry_scene_config_matches_architecture_demo() -> None:
    scene = _load_entry_scene()

    assert scene.scene_id == "entry_auto_light_v1"
    assert scene.context_devices.context_only == [RADAR, LIGHT]
    assert [(pre.device, pre.field, pre.op, pre.value) for pre in scene.preconditions] == [
        (RADAR, "payload.illuminance", "lt", 30),
        (LIGHT, "payload.state", "neq", "ON"),
    ]
    assert scene.actions_hint[0].args == {"device": LIGHT, "action": "turn_on"}


def test_entry_scene_triggers_when_dark_and_light_is_off() -> None:
    hits = _run_entry_sequence(illuminance=12, light_state="OFF")

    assert len(hits) == 1
    assert hits[0].scene_id == "entry_auto_light_v1"
    assert hits[0].facts[RADAR]["payload.illuminance"] == 12
    assert hits[0].facts[LIGHT]["payload.state"] == "OFF"


def test_entry_scene_rejects_bright_environment_before_agent() -> None:
    hits = _run_entry_sequence(illuminance=120, light_state="OFF")

    assert hits == []


def test_entry_scene_rejects_when_light_is_already_on_before_agent() -> None:
    hits = _run_entry_sequence(illuminance=12, light_state="ON")

    assert hits == []
