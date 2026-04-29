from banbu.context.assembler import assemble_blocks
from banbu.context.selector import SelectedContext
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.scenes.definition import Scene
from banbu.state.snapshot_cache import Snapshot
from banbu.turn.model import ProactiveTrigger, Turn


def test_assemble_blocks_include_tool_scene_trigger_feedback_and_device_snapshot() -> None:
    scene = Scene.model_validate(
        {
            "scene_id": "entry_auto_light_v1",
            "name": "Entry auto light",
            "kind": "sequential",
            "trigger": {
                "steps": [
                    {
                        "device": "switch_entry_light",
                        "field": "payload.state",
                        "transition": "OFF->ON",
                    }
                ]
            },
            "context_devices": {
                "trigger": ["switch_entry_light"],
                "context_only": [],
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
    device = ResolvedDevice(
        spec=DeviceSpec(
            friendly_name="switch_entry_light",
            room="玄关",
            role="light_switch",
            aliases=["玄关灯", "入户灯"],
            care_fields=["state"],
        ),
        local_id=12,
        ieee_address="0xlight",
        model="TS011F",
        capabilities={"state"},
    )
    trigger = ProactiveTrigger(
        scene_id=scene.scene_id,
        home_id="home_default",
        facts={"switch_entry_light": {"state": "OFF"}},
        source_event_summaries=["switch_entry_light.state: 'OFF'->'ON'"],
    )
    ctx = SelectedContext(
        turn=Turn.from_proactive(trigger),
        scene=scene,
        devices=[device],
        snapshots={
            "switch_entry_light": Snapshot(
                local_id=12,
                friendly_name="switch_entry_light",
                payload={"state": "OFF"},
            )
        },
    )

    blocks = assemble_blocks(ctx)

    assert blocks[0].startswith("[system policy]")
    assert blocks[1].startswith("[tool schema]")
    assert "[scene:entry_auto_light_v1]" in blocks[2]
    assert "actions_hint: local_id=12 action=turn_on" in blocks[2]
    assert "[trigger] id=" in blocks[3]
    assert blocks[4] == "[feedback]\n  (none)"

    device_block = blocks[5]
    assert device_block.startswith("[device:switch_entry_light] local_id=12")
    assert "room=玄关" in device_block
    assert "role=light_switch" in device_block
    assert 'aliases=["玄关灯", "入户灯"]' in device_block
    assert "actions=['turn_off', 'turn_on']" in device_block
    assert 'snapshot={"state": "OFF"}' in device_block
