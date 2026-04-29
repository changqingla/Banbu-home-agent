from banbu.context.assembler import assemble_blocks
from banbu.context.pilot import optimize
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
                "context_only": ["switch_entry_light"],
            },
            "intent": "Turn on the entry light",
            "actions_hint": [
                {
                    "tool": "execute_plan",
                    "args": {"device": "switch_entry_light", "action": "turn_on"},
                }
            ],
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
            ),
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="switch_entry_light",
                    role="light_switch",
                    care_fields=["state"],
                    actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
                ),
                local_id=12,
                ieee_address="0xlight",
                model="switch",
                capabilities={"state"},
            ),
        ]
    )


def _selected_context():
    scene = _scene()
    resolver = _resolver()
    cache = SnapshotCache(resolver)
    cache.update(4, {"contact": False}, source="test")
    cache.update(12, {"state": "OFF"}, source="test")
    trigger = ProactiveTrigger(
        scene_id=scene.scene_id,
        home_id="home_default",
        facts={"door_sensor_1": {"contact": False}},
        source_event_summaries=["door_sensor_1.contact: True->False"],
    )
    feedback = FeedbackStore()
    feedback.add(FeedbackEntry(
        home_id="home_default",
        scene_id=scene.scene_id,
        trigger_id="trg_previous",
        outcome="success",
        summary="executed 1 action(s)",
    ))
    return select(Turn.from_proactive(trigger), scene, resolver, cache, feedback_store=feedback)


def test_context_blocks_have_stable_order_and_tool_schema() -> None:
    blocks = assemble_blocks(_selected_context())

    assert blocks[0].startswith("[system policy]")
    assert blocks[1].startswith("[tool schema]")
    assert blocks[2].startswith("[scene:entry_auto_light_v1]")
    assert blocks[3].startswith("[trigger]")
    assert blocks[4].startswith("[feedback]")
    assert blocks[5].startswith("[device:")

    assert "execute_plan action array" in blocks[1]
    assert "Skip by returning []" in blocks[1]


def test_scene_hints_and_device_actions_are_visible_to_agent() -> None:
    blocks = assemble_blocks(_selected_context())
    joined = "\n".join(blocks)

    assert "actions_hint: local_id=12 action=turn_on" in joined
    assert "role=light_switch" in joined
    assert "actions=['turn_off', 'turn_on']" in joined
    assert 'snapshot={"state": "OFF"}' in joined


def test_contextpilot_fallback_messages_are_valid(monkeypatch) -> None:
    def broken_optimize(*args, **kwargs):
        raise RuntimeError("contextpilot unavailable")

    monkeypatch.setattr("banbu.context.pilot.cp.optimize", broken_optimize)

    messages = optimize(["[system policy]\npolicy", "[tool schema]\nexecute_plan"], conversation_id="home_scene")

    assert messages == [
        {
            "role": "system",
            "content": "[1] [system policy]\npolicy\n\n[2] [tool schema]\nexecute_plan",
        },
        {
            "role": "user",
            "content": "Scene matched. Output JSON array of actions now.",
        },
    ]
