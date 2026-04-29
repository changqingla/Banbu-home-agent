from banbu.context.assembler import assemble_blocks
from banbu.context.selector import SelectedContext
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import Snapshot
from banbu.turn.model import ProactiveTrigger, Turn


def test_assemble_blocks_includes_scene_trigger_and_device_snapshot() -> None:
    scene = Scene.model_validate(
        {
            "scene_id": "entry_scene",
            "name": "Entry scene",
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": "entry_light",
                        "field": "payload.state",
                        "transition": "OFF->ON",
                    }
                ]
            },
            "context_devices": {"trigger": ["entry_light"], "context_only": []},
            "actions_hint": [
                {
                    "tool": "execute_plan",
                    "args": {"device": "entry_light", "action": "turn_on"},
                }
            ],
        }
    )
    device = ResolvedDevice(
        spec=DeviceSpec(
            friendly_name="entry_light",
            role="light_switch",
            actions={"turn_on": {"state": "ON"}, "turn_off": {"state": "OFF"}},
        ),
        local_id=2,
        ieee_address="0x2",
        model="test",
        capabilities={"state"},
    )
    trigger = ProactiveTrigger(
        scene_id="entry_scene",
        home_id="home_a",
        facts={"entry_light": {"state": "OFF"}},
        source_event_summaries=["entry_light.state: 'OFF'->'ON'"],
    )
    ctx = SelectedContext(
        turn=Turn.from_proactive(trigger),
        scene=scene,
        devices=[device],
        snapshots={
            "entry_light": Snapshot(local_id=2, friendly_name="entry_light", payload={"state": "OFF"})
        },
    )

    blocks = assemble_blocks(ctx)

    assert blocks[0].startswith("[system policy]")
    assert "[scene:entry_scene]" in blocks[1]
    assert "actions_hint: local_id=2 action=turn_on" in blocks[1]
    assert "[trigger] id=" in blocks[2]
    assert blocks[3] == "[device:entry_light] local_id=2 role=light_switch actions=['turn_off', 'turn_on'] snapshot={\"state\": \"OFF\"}"
