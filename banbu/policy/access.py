from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from banbu.devices.definition import ResolvedDevice

Actor = Literal["proactive", "reactive", "system"]


class PolicyLoadError(RuntimeError):
    pass


class AllowRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str
    actions: list[str] = Field(min_length=1)


class ReactiveUserPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    home_id: str
    allowed: list[AllowRule] = Field(default_factory=list)


class SafetyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_risk_roles: list[str] = Field(default_factory=list)
    high_risk_actions: list[str] = Field(default_factory=list)
    proactive_allowed_scenes: list[str] = Field(default_factory=list)


class AccessPolicyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reactive_users: dict[str, ReactiveUserPolicy] = Field(default_factory=dict)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)


@dataclass(frozen=True)
class AccessRequest:
    actor: Actor
    home_id: str | None
    user_id: str | None
    device: ResolvedDevice
    action: str
    scene_id: str | None = None


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


class AccessPolicy:
    def __init__(self, policy_file: AccessPolicyFile) -> None:
        self._policy = policy_file

    def authorize(self, request: AccessRequest) -> AccessDecision:
        if request.actor == "reactive":
            return self._authorize_reactive(request)
        if request.actor in ("proactive", "system"):
            return self._authorize_proactive_or_system(request)
        return AccessDecision(False, f"unknown actor {request.actor!r}")

    def _is_high_risk(self, request: AccessRequest) -> bool:
        return (
            request.device.spec.role in self._policy.safety.high_risk_roles
            or request.action in self._policy.safety.high_risk_actions
        )

    def _authorize_reactive(self, request: AccessRequest) -> AccessDecision:
        if self._is_high_risk(request):
            return AccessDecision(
                False,
                (
                    f"reactive user {request.user_id!r} cannot execute high-risk "
                    f"action {request.action!r} on role {request.device.spec.role!r}"
                ),
            )
        if not request.user_id:
            return AccessDecision(False, "reactive actions require user_id")
        if not request.home_id:
            return AccessDecision(False, "reactive actions require home_id")

        user_policy = self._policy.reactive_users.get(request.user_id)
        if user_policy is None:
            return AccessDecision(False, f"reactive user {request.user_id!r} is not authorized")
        if user_policy.home_id != request.home_id:
            return AccessDecision(
                False,
                (
                    f"reactive user {request.user_id!r} is authorized for home "
                    f"{user_policy.home_id!r}, not {request.home_id!r}"
                ),
            )

        for rule in user_policy.allowed:
            if rule.device == request.device.spec.friendly_name and request.action in rule.actions:
                return AccessDecision(True, "allowed by reactive user policy")

        return AccessDecision(
            False,
            (
                f"reactive user {request.user_id!r} is not allowed to execute "
                f"{request.action!r} on {request.device.spec.friendly_name!r}"
            ),
        )

    def _authorize_proactive_or_system(self, request: AccessRequest) -> AccessDecision:
        if not self._is_high_risk(request):
            return AccessDecision(True, f"{request.actor} action allowed by default safety policy")
        if request.scene_id in self._policy.safety.proactive_allowed_scenes:
            return AccessDecision(True, "high-risk proactive scene allowed by safety policy")
        return AccessDecision(
            False,
            (
                f"high-risk action {request.action!r} on role {request.device.spec.role!r} "
                f"is not allowed for scene {request.scene_id!r}"
            ),
        )


def load_policy(path: Path) -> AccessPolicy:
    if not path.exists():
        raise PolicyLoadError(
            f"policy file not found: {path}. Add banbu/config/policy.yaml with explicit reactive allowlist."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        raise PolicyLoadError(f"policy file is empty: {path}")
    try:
        policy_file = AccessPolicyFile.model_validate(raw)
    except ValidationError as e:
        raise PolicyLoadError(f"policy file failed schema validation: {e}") from e
    return AccessPolicy(policy_file)
