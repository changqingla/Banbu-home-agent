from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class _Wildcard:
    _instance: "_Wildcard | None" = None

    def __new__(cls) -> "_Wildcard":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "*"


WILDCARD = _Wildcard()


def parse_value(s: str) -> Any:
    """Convert a string from a transition spec to a typed value.

    Avoids ``yaml.safe_load`` because YAML 1.1 maps ``ON``/``OFF``/``Yes``/``No``
    to bool, which is wrong for device states like switch ``state: ON``.
    """
    s = s.strip()
    if s == "*":
        return WILDCARD
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


class TriggerStep(BaseModel):
    device: str
    field: str
    transition: str
    within_seconds: float | None = None

    old_value: Any = None
    new_value: Any = None

    @model_validator(mode="after")
    def _parse_transition(self) -> "TriggerStep":
        if "->" not in self.transition:
            raise ValueError(f"transition must be 'old->new', got {self.transition!r}")
        left, right = self.transition.split("->", 1)
        object.__setattr__(self, "old_value", parse_value(left))
        object.__setattr__(self, "new_value", parse_value(right))
        return self


class Trigger(BaseModel):
    steps: list[TriggerStep] = Field(min_length=1)


class VisionTrigger(BaseModel):
    device: str
    field: str = "payload.scene_id"
    value: str
    confidence_field: str = "payload.confidence"
    detected_field: str = "payload.detected"
    frame_id_field: str = "payload.frame_id"


class ContextDevices(BaseModel):
    trigger: list[str] = Field(default_factory=list)
    context_only: list[str] = Field(default_factory=list)


Op = Literal["eq", "neq", "lt", "lte", "gt", "gte", "in"]
OnMissing = Literal["skip", "pass", "fail"]


class Precondition(BaseModel):
    device: str
    field: str
    op: Op
    value: Any
    on_missing: OnMissing = "skip"


class ActionsHint(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Policy(BaseModel):
    cooldown_seconds: float = 60.0
    inflight_seconds: float = 30.0
    priority: int = 5


class VisionPolicy(BaseModel):
    confidence_threshold: float = 0.7
    consecutive_hits: int = Field(default=2, ge=1)
    reset_on_miss: bool = True


class Scene(BaseModel):
    scene_id: str
    name: str
    kind: Literal["sequential", "vision_match"] = "sequential"
    trigger: Trigger | VisionTrigger
    vision_policy: VisionPolicy = Field(default_factory=VisionPolicy)
    vision_criteria: list[str] = Field(default_factory=list)
    context_devices: ContextDevices = Field(default_factory=ContextDevices)
    preconditions: list[Precondition] = Field(default_factory=list)
    intent: str = ""
    actions_hint: list[ActionsHint] = Field(default_factory=list)
    policy: Policy = Field(default_factory=Policy)

    @field_validator("scene_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError("scene_id must be a non-empty token without spaces")
        return v

    @model_validator(mode="after")
    def _trigger_matches_kind(self) -> "Scene":
        if self.kind == "sequential" and not isinstance(self.trigger, Trigger):
            raise ValueError("sequential scenes require trigger.steps")
        if self.kind == "vision_match" and not isinstance(self.trigger, VisionTrigger):
            raise ValueError("vision_match scenes require trigger.device/field/value")
        return self

    def trigger_devices(self) -> set[str]:
        if isinstance(self.trigger, VisionTrigger):
            names: set[str] = {self.trigger.device}
        else:
            names = {step.device for step in self.trigger.steps}
        names.update(self.context_devices.trigger)
        return names

    def all_referenced_devices(self) -> set[str]:
        names: set[str] = self.trigger_devices() | set(self.context_devices.context_only)
        names.update(p.device for p in self.preconditions)
        names.update(h.args["device"] for h in self.actions_hint if "device" in h.args)
        return names
