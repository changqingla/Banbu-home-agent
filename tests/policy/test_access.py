from pathlib import Path

import pytest

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.policy.access import AccessPolicy, AccessPolicyFile, AccessRequest, PolicyLoadError, load_policy


def _device(
    friendly_name: str,
    *,
    local_id: int,
    role: str,
) -> ResolvedDevice:
    return ResolvedDevice(
        spec=DeviceSpec(friendly_name=friendly_name, role=role),
        local_id=local_id,
        ieee_address=f"0x{local_id}",
        model="test",
        capabilities={"state", "warning"},
    )


def _policy() -> AccessPolicy:
    return AccessPolicy(
        AccessPolicyFile.model_validate(
            {
                "reactive_users": {
                    "user_1": {
                        "home_id": "home_a",
                        "allowed": [
                            {
                                "device": "switch_entry_light",
                                "actions": ["turn_on", "turn_off"],
                            }
                        ],
                    }
                },
                "safety": {
                    "high_risk_roles": ["siren"],
                    "high_risk_actions": ["alarm_burglar", "alarm_fire"],
                    "proactive_allowed_scenes": ["safety_smoke_then_gas_v1"],
                },
            }
        )
    )


def test_reactive_allowlist_authorizes_matching_user_home_device_action() -> None:
    policy = _policy()
    decision = policy.authorize(
        AccessRequest(
            actor="reactive",
            home_id="home_a",
            user_id="user_1",
            device=_device("switch_entry_light", local_id=2, role="light_switch"),
            action="turn_on",
        )
    )

    assert decision.allowed is True


def test_reactive_unknown_user_is_denied() -> None:
    policy = _policy()
    decision = policy.authorize(
        AccessRequest(
            actor="reactive",
            home_id="home_a",
            user_id="stranger",
            device=_device("switch_entry_light", local_id=2, role="light_switch"),
            action="turn_on",
        )
    )

    assert decision.allowed is False
    assert "not authorized" in decision.reason


def test_reactive_high_risk_action_is_denied_before_allowlist() -> None:
    policy = _policy()
    decision = policy.authorize(
        AccessRequest(
            actor="reactive",
            home_id="home_a",
            user_id="user_1",
            device=_device("entry_siren", local_id=6, role="siren"),
            action="alarm_fire",
        )
    )

    assert decision.allowed is False
    assert "high-risk" in decision.reason


def test_proactive_high_risk_scene_must_be_allowed() -> None:
    policy = _policy()
    siren = _device("entry_siren", local_id=6, role="siren")

    denied = policy.authorize(
        AccessRequest(
            actor="proactive",
            home_id="home_a",
            user_id=None,
            device=siren,
            action="alarm_fire",
            scene_id="entry_auto_light_v1",
        )
    )
    allowed = policy.authorize(
        AccessRequest(
            actor="proactive",
            home_id="home_a",
            user_id=None,
            device=siren,
            action="alarm_fire",
            scene_id="safety_smoke_then_gas_v1",
        )
    )

    assert denied.allowed is False
    assert allowed.allowed is True


def test_load_policy_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PolicyLoadError):
        load_policy(tmp_path / "missing.yaml")
