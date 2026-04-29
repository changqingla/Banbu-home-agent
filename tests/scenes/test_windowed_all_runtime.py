from __future__ import annotations

from pathlib import Path

import yaml

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.dispatcher import Dispatcher
from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.scenes.definition import Scene
from banbu.scenes.reverse_index import build_reverse_index
from banbu.scenes.runtime.windowed_all import WindowedAllSceneRuntime
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import ProactiveTrigger


REPO_ROOT = Path(__file__).resolve().parents[2]

SMOKE = "smoke_detector_1"
GAS = "gas_sensor_1"
SIREN = "siren_1"
LIGHT = "light_1"


class ManualClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=SMOKE, role="smoke_detector", care_fields=["smoke"]),
                local_id=11,
                ieee_address="0xsmoke",
                model="smoke",
                capabilities={"smoke"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=GAS, role="gas_sensor", care_fields=["gas"]),
                local_id=9,
                ieee_address="0xgas",
                model="gas",
                capabilities={"gas"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=SIREN, role="siren", care_fields=[]),
                local_id=6,
                ieee_address="0xsiren",
                model="siren",
                capabilities={"warning"},
            ),
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=LIGHT, role="light_switch", care_fields=["state"]),
                local_id=12,
                ieee_address="0xlight",
                model="switch",
                capabilities={"state"},
            ),
        ]
    )


def _cache(*, light_state: str = "OFF") -> SnapshotCache:
    cache = SnapshotCache(_resolver())
    cache.update(11, {"smoke": True}, source="test")
    cache.update(9, {"gas": True}, source="test")
    cache.update(6, {"warning": {"mode": "stop"}}, source="test")
    cache.update(12, {"state": light_state}, source="test")
    return cache


def _scene(*, with_precondition: bool = False) -> Scene:
    raw = {
        "scene_id": "safety_smoke_gas_windowed_v1",
        "name": "Smoke and gas in any order",
        "kind": "windowed_all",
        "trigger": {
            "window_seconds": 10,
            "conditions": [
                {
                    "device": SMOKE,
                    "field": "payload.smoke",
                    "transition": "false->true",
                },
                {
                    "device": GAS,
                    "field": "payload.gas",
                    "transition": "false->true",
                },
            ],
        },
        "context_devices": {
            "trigger": [SMOKE, GAS],
            "context_only": [LIGHT] if with_precondition else [],
        },
        "preconditions": [],
        "intent": "Fire alarm when smoke and gas both trigger within 10s",
        "actions_hint": [
            {
                "tool": "execute_plan",
                "args": {"device": SIREN, "action": "alarm_fire"},
            }
        ],
        "policy": {
            "cooldown_seconds": 300,
            "inflight_seconds": 60,
            "priority": 10,
        },
    }
    if with_precondition:
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
    friendly_name: str,
    local_id: int,
    ieee_address: str,
    field: str,
) -> tuple[DeviceEvent, FieldChange]:
    change = FieldChange(field=field, old=False, new=True)
    event = DeviceEvent(
        local_id=local_id,
        friendly_name=friendly_name,
        ieee_address=ieee_address,
        payload={field: True},
        changes=[change],
    )
    return event, change


def _smoke_event() -> tuple[DeviceEvent, FieldChange]:
    return _event(friendly_name=SMOKE, local_id=11, ieee_address="0xsmoke", field="smoke")


def _gas_event() -> tuple[DeviceEvent, FieldChange]:
    return _event(friendly_name=GAS, local_id=9, ieee_address="0xgas", field="gas")


def _runtime(
    scene: Scene | None = None,
    *,
    cache: SnapshotCache | None = None,
    clock: ManualClock | None = None,
) -> tuple[WindowedAllSceneRuntime, list[ProactiveTrigger], ManualClock]:
    hits: list[ProactiveTrigger] = []
    manual_clock = clock or ManualClock()
    runtime = WindowedAllSceneRuntime(
        scene or _scene(),
        cache or _cache(),
        home_id="home_default",
        on_hit=hits.append,
        clock=manual_clock,
    )
    return runtime, hits, manual_clock


def test_windowed_all_triggers_gas_then_smoke_within_window() -> None:
    runtime, hits, clock = _runtime()
    gas_event, gas_change = _gas_event()
    smoke_event, smoke_change = _smoke_event()

    runtime.on_event(gas_event, gas_change)
    clock.advance(4)
    runtime.on_event(smoke_event, smoke_change)

    assert len(hits) == 1
    assert hits[0].source_event_summaries == [
        "smoke_detector_1.smoke: False->True",
        "gas_sensor_1.gas: False->True",
    ]
    assert len(hits[0].source_event_ids) == 2
    assert all(eid.startswith("evt_") for eid in hits[0].source_event_ids)
    assert hits[0].facts[SMOKE] == {"smoke": True}
    assert hits[0].facts[GAS] == {"gas": True}


def test_windowed_all_triggers_smoke_then_gas_within_window() -> None:
    runtime, hits, clock = _runtime()
    smoke_event, smoke_change = _smoke_event()
    gas_event, gas_change = _gas_event()

    runtime.on_event(smoke_event, smoke_change)
    clock.advance(4)
    runtime.on_event(gas_event, gas_change)

    assert len(hits) == 1


def test_windowed_all_does_not_trigger_when_events_are_outside_window() -> None:
    runtime, hits, clock = _runtime()
    smoke_event, smoke_change = _smoke_event()
    gas_event, gas_change = _gas_event()

    runtime.on_event(smoke_event, smoke_change)
    clock.advance(11)
    runtime.on_event(gas_event, gas_change)

    assert hits == []


def test_windowed_all_preconditions_accept_or_reject_hit() -> None:
    smoke_event, smoke_change = _smoke_event()
    gas_event, gas_change = _gas_event()

    accepted, accepted_hits, accepted_clock = _runtime(
        _scene(with_precondition=True),
        cache=_cache(light_state="OFF"),
    )
    accepted.on_event(smoke_event, smoke_change)
    accepted_clock.advance(1)
    accepted.on_event(gas_event, gas_change)
    assert len(accepted_hits) == 1
    assert accepted_hits[0].facts[LIGHT]["payload.state"] == "OFF"

    rejected, rejected_hits, rejected_clock = _runtime(
        _scene(with_precondition=True),
        cache=_cache(light_state="ON"),
    )
    rejected.on_event(smoke_event, smoke_change)
    rejected_clock.advance(1)
    rejected.on_event(gas_event, gas_change)
    assert rejected_hits == []


def test_windowed_all_respects_cooldown_and_inflight() -> None:
    smoke_event, smoke_change = _smoke_event()
    gas_event, gas_change = _gas_event()

    cooling, cooling_hits, cooling_clock = _runtime()
    cooling.state.set_cooldown(60, now=cooling_clock.now)
    cooling.on_event(smoke_event, smoke_change)
    cooling_clock.advance(1)
    cooling.on_event(gas_event, gas_change)
    assert cooling_hits == []

    inflight, inflight_hits, inflight_clock = _runtime()
    inflight.on_event(smoke_event, smoke_change)
    inflight_clock.advance(1)
    inflight.on_event(gas_event, gas_change)
    inflight_clock.advance(1)
    inflight.on_event(smoke_event, smoke_change)
    inflight_clock.advance(1)
    inflight.on_event(gas_event, gas_change)
    assert len(inflight_hits) == 1


def test_dispatcher_wires_windowed_all_runtime_and_reverse_index() -> None:
    scene = _scene()
    resolver = _resolver()
    reverse_index = build_reverse_index([scene], resolver)
    dispatcher = Dispatcher([scene], reverse_index, _cache(), home_id="home_default")

    assert reverse_index.lookup(SMOKE, "payload.smoke") == [
        ("safety_smoke_gas_windowed_v1", "trigger")
    ]
    assert reverse_index.lookup(GAS, "payload.gas") == [
        ("safety_smoke_gas_windowed_v1", "trigger")
    ]
    assert isinstance(dispatcher.runtime("safety_smoke_gas_windowed_v1"), WindowedAllSceneRuntime)


def test_tracked_safety_scenes_are_replaced_by_one_windowed_scene() -> None:
    scenes_dir = REPO_ROOT / "banbu/config/scenes"
    scene_ids = [
        yaml.safe_load(path.read_text(encoding="utf-8"))["scene_id"]
        for path in scenes_dir.glob("safety_*.yaml")
    ]

    assert "safety_smoke_gas_windowed_v1" in scene_ids
    assert "safety_gas_then_smoke_v1" not in scene_ids
    assert "safety_smoke_then_gas_v1" not in scene_ids

    raw = yaml.safe_load((scenes_dir / "safety_smoke_gas_windowed_v1.yaml").read_text(encoding="utf-8"))
    scene = Scene.model_validate(raw)
    assert scene.kind == "windowed_all"
