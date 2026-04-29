from pathlib import Path

import yaml

from banbu.devices.definition import DevicesFile
from banbu.devices.definition import DeviceSpec, effective_actions


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tracked_devices_example_preserves_semantic_fields() -> None:
    raw = yaml.safe_load((REPO_ROOT / "banbu/config/devices.yaml.example").read_text(encoding="utf-8"))

    devices = DevicesFile.model_validate(raw).devices
    entry_light = next(device for device in devices if device.friendly_name == "switch_entry_light")

    assert entry_light.room == "玄关"
    assert entry_light.aliases == ["玄关灯", "入户灯"]
    assert effective_actions(entry_light)["turn_on"] == {"state": "ON"}


def test_device_spec_preserves_semantic_metadata() -> None:
    spec = DeviceSpec(
        friendly_name="switch_entry_light",
        room="玄关",
        role="light_switch",
        aliases=["玄关灯", "入户灯"],
        care_fields=["state"],
    )

    assert spec.room == "玄关"
    assert spec.aliases == ["玄关灯", "入户灯"]


def test_common_switch_roles_have_default_actions() -> None:
    light = DeviceSpec(friendly_name="light", role="light_switch")
    plug = DeviceSpec(friendly_name="plug", role="smart_plug")

    assert effective_actions(light) == {
        "turn_on": {"state": "ON"},
        "turn_off": {"state": "OFF"},
    }
    assert effective_actions(plug) == {
        "turn_on": {"state": "ON"},
        "turn_off": {"state": "OFF"},
    }


def test_color_temperature_light_has_default_light_controls() -> None:
    spec = DeviceSpec(friendly_name="color_light", role="color_temp_light")

    actions = effective_actions(spec)

    assert actions["turn_on"] == {"state": "ON"}
    assert actions["turn_off"] == {"state": "OFF"}
    assert actions["brightness_high"] == {"brightness": 254}
    assert actions["color_temp_warm"] == {"color_temp": 454}


def test_explicit_actions_override_and_extend_role_defaults() -> None:
    spec = DeviceSpec(
        friendly_name="light",
        role="light_switch",
        actions={
            "turn_on": {"state": "ON", "brightness": 200},
            "blink": {"effect": "blink"},
        },
    )

    actions = effective_actions(spec)

    assert actions["turn_on"] == {"state": "ON", "brightness": 200}
    assert actions["turn_off"] == {"state": "OFF"}
    assert actions["blink"] == {"effect": "blink"}


def test_effective_actions_returns_independent_payloads() -> None:
    spec = DeviceSpec(friendly_name="light", role="light_switch")

    first = effective_actions(spec)
    first["turn_on"]["state"] = "BROKEN"

    assert effective_actions(spec)["turn_on"] == {"state": "ON"}
