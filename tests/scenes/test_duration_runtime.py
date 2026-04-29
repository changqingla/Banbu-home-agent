from __future__ import annotations

import pytest

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.dispatcher import Dispatcher
from banbu.ingest.event import DeviceEvent, FieldChange
from banbu.ingest.poller import FallbackPoller
from banbu.scenes.definition import Scene
from banbu.scenes.reverse_index import build_reverse_index
from banbu.scenes.runtime.duration import DurationSceneRuntime
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import ProactiveTrigger


RADAR = "presence_radar_1"
LIGHT = "light_1"


class ManualClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class AllInfoClient:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    async def get_allinfo(self) -> list[dict]:
        return self.entries


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name=RADAR, role="presence_radar", care_fields=["presence"]),
                local_id=16,
                ieee_address="0xradar",
                model="radar",
                capabilities={"presence"},
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


def _cache(*, presence: bool = False, light_state: str = "OFF") -> SnapshotCache:
    cache = SnapshotCache(_resolver())
    cache.update(16, {"presence": presence}, source="test")
    cache.update(12, {"state": light_state}, source="test")
    return cache


def _scene(*, with_precondition: bool = False) -> Scene:
    raw = {
        "scene_id": "no_presence_for_10s",
        "name": "No presence for 10 seconds",
        "kind": "duration_triggered",
        "trigger": {
            "duration_seconds": 10,
            "condition": {
                "device": RADAR,
                "field": "payload.presence",
                "value": False,
            },
        },
        "context_devices": {
            "trigger": [RADAR],
            "context_only": [LIGHT] if with_precondition else [],
        },
        "preconditions": [],
        "intent": "Turn off the light when no one is present for 10 seconds",
        "actions_hint": [
            {
                "tool": "execute_plan",
                "args": {"device": LIGHT, "action": "turn_off"},
            }
        ],
        "policy": {
            "cooldown_seconds": 60,
            "inflight_seconds": 30,
            "priority": 5,
        },
    }
    if with_precondition:
        raw["preconditions"] = [
            {
                "device": LIGHT,
                "field": "payload.state",
                "op": "eq",
                "value": "ON",
                "on_missing": "skip",
            }
        ]
    return Scene.model_validate(raw)


def _presence_event(new: bool) -> tuple[DeviceEvent, FieldChange]:
    change = FieldChange(field="presence", old=not new, new=new)
    event = DeviceEvent(
        local_id=16,
        friendly_name=RADAR,
        ieee_address="0xradar",
        payload={"presence": new},
        changes=[change],
    )
    return event, change


def _runtime(
    scene: Scene | None = None,
    *,
    cache: SnapshotCache | None = None,
    clock: ManualClock | None = None,
) -> tuple[DurationSceneRuntime, list[ProactiveTrigger], ManualClock]:
    hits: list[ProactiveTrigger] = []
    manual_clock = clock or ManualClock()
    runtime = DurationSceneRuntime(
        scene or _scene(),
        cache or _cache(),
        home_id="home_default",
        on_hit=hits.append,
        clock=manual_clock,
    )
    return runtime, hits, manual_clock


def test_duration_trigger_emits_after_condition_held_for_duration() -> None:
    runtime, hits, clock = _runtime()

    runtime.on_tick()
    clock.advance(9)
    runtime.on_tick()
    assert hits == []

    clock.advance(1)
    runtime.on_tick()

    assert len(hits) == 1
    assert hits[0].scene_id == "no_presence_for_10s"
    assert hits[0].facts[RADAR] == {"presence": False}
    assert hits[0].facts["duration"]["required_seconds"] == 10
    assert hits[0].source_event_summaries == [
        "presence_radar_1.presence held False for 10.0s"
    ]


def test_duration_trigger_resets_when_condition_becomes_false() -> None:
    cache = _cache(presence=False)
    runtime, hits, clock = _runtime(cache=cache)

    runtime.on_tick()
    clock.advance(6)
    cache.update(16, {"presence": True}, source="test")
    event, change = _presence_event(True)
    runtime.on_event(event, change)
    clock.advance(10)
    runtime.on_tick()

    assert hits == []
    assert runtime.condition_satisfied_since is None


def test_duration_trigger_starts_from_event_and_completes_on_tick() -> None:
    cache = _cache(presence=True)
    runtime, hits, clock = _runtime(cache=cache)

    cache.update(16, {"presence": False}, source="test")
    event, change = _presence_event(False)
    runtime.on_event(event, change)
    clock.advance(10)
    runtime.on_tick()

    assert len(hits) == 1


def test_duration_trigger_preconditions_accept_or_reject_hit() -> None:
    accepted, accepted_hits, accepted_clock = _runtime(
        _scene(with_precondition=True),
        cache=_cache(presence=False, light_state="ON"),
    )
    accepted.on_tick()
    accepted_clock.advance(10)
    accepted.on_tick()
    assert len(accepted_hits) == 1
    assert accepted_hits[0].facts[LIGHT]["payload.state"] == "ON"

    rejected, rejected_hits, rejected_clock = _runtime(
        _scene(with_precondition=True),
        cache=_cache(presence=False, light_state="OFF"),
    )
    rejected.on_tick()
    rejected_clock.advance(10)
    rejected.on_tick()
    assert rejected_hits == []


def test_duration_trigger_respects_cooldown_and_inflight() -> None:
    cooling, cooling_hits, cooling_clock = _runtime()
    cooling.state.set_cooldown(60, now=cooling_clock.now)
    cooling.on_tick()
    cooling_clock.advance(10)
    cooling.on_tick()
    assert cooling_hits == []

    inflight, inflight_hits, inflight_clock = _runtime()
    inflight.state.set_inflight(60, now=inflight_clock.now)
    inflight.on_tick()
    inflight_clock.advance(10)
    inflight.on_tick()
    assert inflight_hits == []


def test_dispatcher_wires_duration_runtime_and_tick() -> None:
    scene = _scene()
    resolver = _resolver()
    hits: list[ProactiveTrigger] = []
    dispatcher = Dispatcher(
        [scene],
        build_reverse_index([scene], resolver),
        _cache(),
        home_id="home_default",
        on_hit=hits.append,
    )

    assert build_reverse_index([scene], resolver).lookup(RADAR, "payload.presence") == [
        ("no_presence_for_10s", "trigger")
    ]
    assert isinstance(dispatcher.runtime("no_presence_for_10s"), DurationSceneRuntime)
    dispatcher.on_tick()


@pytest.mark.asyncio
async def test_fallback_poller_invokes_tick_even_without_payload_changes() -> None:
    resolver = _resolver()
    cache = _cache(presence=False)
    ticks = 0

    def on_tick() -> None:
        nonlocal ticks
        ticks += 1

    poller = FallbackPoller(
        AllInfoClient([{"local_id": 16, "payload": {"presence": False}}]),
        resolver,
        cache,
        interval_seconds=30,
        on_tick=on_tick,
    )

    await poller._tick()

    assert ticks == 1
