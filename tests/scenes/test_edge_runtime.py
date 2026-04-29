from __future__ import annotations

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.dispatcher import Dispatcher
from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene
from banbu.scenes.reverse_index import build_reverse_index
from banbu.scenes.runtime.edge import EdgeSceneRuntime
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import ProactiveTrigger


DOOR = "door_sensor_1"
PIR = "entry_pir_1"
LIGHT = "switch_entry_light"


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=DOOR, role="door_sensor", care_fields=["contact"]),
                local_id=1,
                ieee_address="0xdoor",
                model="door",
                capabilities={"contact"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=PIR, role="motion_sensor", care_fields=["occupancy"]),
                local_id=2,
                ieee_address="0xpir",
                model="pir",
                capabilities={"occupancy"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name=LIGHT,
                    role="light_switch",
                    care_fields=["state"],
                    actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
                ),
                local_id=3,
                ieee_address="0xlight",
                model="switch",
                capabilities={"state"},
            ),
        ]
    )


def _cache(*, light_state: str = "OFF") -> SnapshotCache:
    cache = SnapshotCache(_resolver())
    cache.update(1, {"contact": False}, source="test")
    cache.update(2, {"occupancy": True}, source="test")
    cache.update(3, {"state": light_state}, source="test")
    return cache


def _scene(
    *,
    device: str = DOOR,
    field: str = "payload.contact",
    transition: str = "true->false",
    with_light_precondition: bool = False,
) -> Scene:
    raw = {
        "scene_id": "edge_scene",
        "name": "Edge scene",
        "kind": "edge_triggered",
        "trigger": {
            "steps": [
                {
                    "device": device,
                    "field": field,
                    "transition": transition,
                }
            ]
        },
        "context_devices": {
            "trigger": [device],
            "context_only": [LIGHT] if with_light_precondition else [],
        },
        "preconditions": [],
        "intent": "Run on one edge",
        "actions_hint": [],
        "policy": {
            "cooldown_seconds": 60,
            "inflight_seconds": 30,
            "priority": 5,
        },
    }
    if with_light_precondition:
        raw["preconditions"] = [
            {
                "device": LIGHT,
                "field": "payload.state",
                "op": "neq",
                "value": "ON",
                "on_missing": "skip",
            }
        ]
    return Scene.model_validate(raw)


def _event(
    *,
    local_id: int = 1,
    friendly_name: str = DOOR,
    ieee_address: str = "0xdoor",
    field: str = "contact",
    old: object = True,
    new: object = False,
) -> tuple[DeviceEvent, FieldChange]:
    change = FieldChange(field=field, old=old, new=new)
    event = DeviceEvent(
        local_id=local_id,
        friendly_name=friendly_name,
        ieee_address=ieee_address,
        payload={field: new},
        changes=[change],
    )
    return event, change


def _runtime(scene: Scene, *, cache: SnapshotCache | None = None) -> tuple[EdgeSceneRuntime, list[ProactiveTrigger]]:
    hits: list[ProactiveTrigger] = []
    runtime = EdgeSceneRuntime(
        scene,
        cache or _cache(),
        home_id="home_default",
        on_hit=hits.append,
    )
    return runtime, hits


def test_edge_trigger_emits_proactive_trigger_on_single_transition() -> None:
    runtime, hits = _runtime(_scene())
    event, change = _event()

    runtime.on_event(event, change)

    assert len(hits) == 1
    assert hits[0].scene_id == "edge_scene"
    assert hits[0].facts[DOOR] == {"contact": False}
    assert hits[0].source_event_summaries == ["door_sensor_1.contact: True->False"]
    assert len(hits[0].source_event_ids) == 1
    assert hits[0].source_event_ids[0].startswith("evt_")
    assert runtime.state.is_inflight()


def test_edge_trigger_ignores_non_matching_transition() -> None:
    runtime, hits = _runtime(_scene())
    event, change = _event(old=False, new=True)

    runtime.on_event(event, change)

    assert hits == []


def test_edge_trigger_supports_wildcard_transition() -> None:
    runtime, hits = _runtime(_scene(device=PIR, field="payload.occupancy", transition="*->true"))
    event, change = _event(
        local_id=2,
        friendly_name=PIR,
        ieee_address="0xpir",
        field="occupancy",
        old=False,
        new=True,
    )

    runtime.on_event(event, change)

    assert len(hits) == 1
    assert hits[0].facts[PIR] == {"occupancy": True}


def test_edge_trigger_preconditions_accept_or_reject_hit() -> None:
    event, change = _event()

    accepted, accepted_hits = _runtime(
        _scene(with_light_precondition=True),
        cache=_cache(light_state="OFF"),
    )
    accepted.on_event(event, change)
    assert len(accepted_hits) == 1
    assert accepted_hits[0].facts[LIGHT]["payload.state"] == "OFF"

    rejected, rejected_hits = _runtime(
        _scene(with_light_precondition=True),
        cache=_cache(light_state="ON"),
    )
    rejected.on_event(event, change)
    assert rejected_hits == []


def test_edge_trigger_respects_cooldown_and_inflight() -> None:
    event, change = _event()

    cooling, cooling_hits = _runtime(_scene())
    cooling.state.set_cooldown(60)
    cooling.on_event(event, change)
    assert cooling_hits == []

    inflight, inflight_hits = _runtime(_scene())
    inflight.on_event(event, change)
    inflight.on_event(event, change)
    assert len(inflight_hits) == 1


def test_dispatcher_wires_edge_runtime() -> None:
    scene = _scene()
    resolver = _resolver()
    dispatcher = Dispatcher(
        [scene],
        build_reverse_index([scene], resolver),
        _cache(),
        home_id="home_default",
    )

    assert isinstance(dispatcher.runtime("edge_scene"), EdgeSceneRuntime)
