import pytest

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.reactive.matcher import ReactiveMatchError, match_device_action


def _device(
    friendly_name: str,
    *,
    local_id: int,
    role: str,
    room: str | None = None,
    aliases: list[str] | None = None,
    actions: dict | None = None,
    capabilities: set[str] | None = None,
) -> ResolvedDevice:
    return ResolvedDevice(
        spec=DeviceSpec(
            friendly_name=friendly_name,
            room=room,
            role=role,
            aliases=aliases or [],
            actions=actions or {},
        ),
        local_id=local_id,
        ieee_address=f"0x{local_id}",
        model="test",
        capabilities=capabilities or {"state"},
    )


def test_matches_turn_on_by_alias() -> None:
    resolver = DeviceResolver(
        [
            _device(
                "switch_entry_light",
                local_id=2,
                room="玄关",
                role="light_switch",
                aliases=["玄关灯", "入户灯"],
            )
        ]
    )

    match = match_device_action("打开玄关灯", resolver)

    assert match.local_id == 2
    assert match.action == "turn_on"
    assert match.device.spec.friendly_name == "switch_entry_light"


def test_matches_turn_off_by_room_and_role_without_alias() -> None:
    resolver = DeviceResolver(
        [
            _device(
                "switch_entry_light",
                local_id=2,
                room="玄关",
                role="light_switch",
            )
        ]
    )

    match = match_device_action("关闭玄关灯", resolver)

    assert match.local_id == 2
    assert match.action == "turn_off"
    assert any(reason.startswith("room+role:") for reason in match.device_reasons)


def test_unknown_device_does_not_select_role_only_match() -> None:
    resolver = DeviceResolver(
        [
            _device(
                "switch_entry_light",
                local_id=2,
                room="玄关",
                role="light_switch",
                aliases=["玄关灯"],
            )
        ]
    )

    with pytest.raises(ReactiveMatchError) as exc:
        match_device_action("打开厨房灯", resolver)

    assert exc.value.kind == "unknown_device"


def test_unsupported_action_is_reported_for_matched_device() -> None:
    resolver = DeviceResolver(
        [
            _device(
                "smoke_alarm",
                local_id=11,
                room="厨房",
                role="smoke_detector",
                aliases=["烟雾报警器"],
                actions={"silence": {"silence": True}},
                capabilities={"silence"},
            )
        ]
    )

    with pytest.raises(ReactiveMatchError) as exc:
        match_device_action("打开烟雾报警器", resolver)

    assert exc.value.kind == "unsupported_action"
    assert exc.value.candidates == ("silence",)


def test_ambiguous_device_match_fails_closed() -> None:
    resolver = DeviceResolver(
        [
            _device(
                "switch_entry_light_a",
                local_id=2,
                room="玄关",
                role="light_switch",
                aliases=["玄关灯"],
            ),
            _device(
                "switch_entry_light_b",
                local_id=3,
                room="玄关",
                role="light_switch",
                aliases=["玄关灯"],
            ),
        ]
    )

    with pytest.raises(ReactiveMatchError) as exc:
        match_device_action("打开玄关灯", resolver)

    assert exc.value.kind == "ambiguous_device"
    assert exc.value.candidates == ("switch_entry_light_a", "switch_entry_light_b")
