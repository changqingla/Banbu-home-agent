from banbu.context.assembler import assemble_blocks
from banbu.context.selector import select
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene
from banbu.state.feedback import FeedbackEntry, FeedbackStore
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.model import ProactiveTrigger, Turn


def _scene() -> Scene:
    return Scene.model_validate(
        {
            "scene_id": "entry_auto_light_v1",
            "name": "Entry auto light",
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": "door_sensor_1",
                        "field": "payload.contact",
                        "transition": "true->false",
                    }
                ]
            },
            "context_devices": {
                "trigger": ["door_sensor_1"],
                "context_only": [],
            },
            "intent": "Turn on entry light",
            "actions_hint": [],
        }
    )


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="door_sensor_1", role="door_sensor", care_fields=["contact"]),
                local_id=4,
                ieee_address="0xdoor",
                model="door",
                capabilities={"contact"},
            )
        ]
    )


def test_context_selection_and_assembly_include_recent_feedback() -> None:
    scene = _scene()
    resolver = _resolver()
    cache = SnapshotCache(resolver)
    cache.update(4, {"contact": False}, source="test")

    trigger = ProactiveTrigger(
        scene_id=scene.scene_id,
        home_id="home_default",
        facts={"door_sensor_1": {"contact": False}},
        source_event_summaries=["door_sensor_1.contact: True->False"],
    )
    turn = Turn.from_proactive(trigger)

    feedback = FeedbackStore()
    feedback.add(FeedbackEntry(
        home_id="home_default",
        scene_id=scene.scene_id,
        trigger_id="trg_previous",
        outcome="failure",
        summary="execute failed",
        details={"error": "device offline"},
    ))

    ctx = select(turn, scene, resolver, cache, feedback_store=feedback)
    blocks = assemble_blocks(ctx)
    feedback_block = next(block for block in blocks if block.startswith("[feedback]"))

    assert ctx.feedback[0].trigger_id == "trg_previous"
    assert '"outcome": "failure"' in feedback_block
    assert '"summary": "execute failed"' in feedback_block
    assert '"error": "device offline"' in feedback_block


def test_context_assembly_includes_empty_feedback_block() -> None:
    scene = _scene()
    resolver = _resolver()
    cache = SnapshotCache(resolver)
    cache.update(4, {"contact": False}, source="test")
    trigger = ProactiveTrigger(scene_id=scene.scene_id, home_id="home_default", facts={})

    ctx = select(Turn.from_proactive(trigger), scene, resolver, cache)

    assert "[feedback]\n  (none)" in assemble_blocks(ctx)
