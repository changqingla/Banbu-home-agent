from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from banbu.devices.definition import ResolvedDevice, effective_actions
from banbu.devices.resolver import DeviceResolver

MatchErrorKind = Literal[
    "unknown_action",
    "ambiguous_action",
    "unknown_device",
    "ambiguous_device",
    "unsupported_action",
]


ACTION_TERMS: dict[str, tuple[str, ...]] = {
    "turn_on": ("turn_on", "turn on", "打开", "开启", "开灯", "点亮"),
    "turn_off": ("turn_off", "turn off", "关闭", "关掉", "关上", "熄灭"),
}

ROLE_NOUNS: dict[str, tuple[str, ...]] = {
    "light_switch": ("灯开关", "灯", "开关", "light", "lamp"),
    "color_temp_light": ("色温灯", "灯泡", "灯", "light", "lamp"),
    "smart_plug": ("智能插座", "插座", "plug", "outlet"),
    "siren": ("报警器", "警报器", "alarm", "siren"),
    "smoke_detector": ("烟雾报警器", "烟感", "smoke"),
    "gas_sensor": ("燃气报警器", "气体传感器", "gas"),
}


@dataclass(frozen=True)
class _ActionHit:
    action: str
    score: int
    terms: tuple[str, ...]


@dataclass(frozen=True)
class _DeviceHit:
    device: ResolvedDevice
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReactiveMatch:
    device: ResolvedDevice
    action: str
    action_terms: tuple[str, ...]
    device_reasons: tuple[str, ...]

    @property
    def local_id(self) -> int:
        return self.device.local_id


class ReactiveMatchError(ValueError):
    def __init__(
        self,
        kind: MatchErrorKind,
        message: str,
        *,
        candidates: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.candidates = candidates


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def _contains(text: str, compact_text: str, term: str) -> bool:
    term = term.strip()
    if not term:
        return False
    folded = term.casefold()
    return folded in text or _compact(folded) in compact_text


def _action_hits(utterance: str) -> list[_ActionHit]:
    text = utterance.casefold()
    compact_text = _compact(utterance)
    hits: list[_ActionHit] = []

    for action, terms in ACTION_TERMS.items():
        matched = tuple(term for term in terms if _contains(text, compact_text, term))
        if matched:
            hits.append(_ActionHit(action=action, score=max(len(_compact(t)) for t in matched), terms=matched))

    return sorted(hits, key=lambda h: (-h.score, h.action))


def _select_action(utterance: str) -> _ActionHit:
    hits = _action_hits(utterance)
    if not hits:
        raise ReactiveMatchError(
            "unknown_action",
            "could not identify a supported action in the utterance",
        )

    top_score = hits[0].score
    top = [hit for hit in hits if hit.score == top_score]
    if len(top) > 1:
        raise ReactiveMatchError(
            "ambiguous_action",
            "utterance contains multiple equally likely actions",
            candidates=tuple(hit.action for hit in top),
        )
    return hits[0]


def _device_terms(device: ResolvedDevice) -> tuple[tuple[str, int, str], ...]:
    spec = device.spec
    terms: list[tuple[str, int, str]] = [(spec.friendly_name, 120, "friendly_name")]
    terms.extend((alias, 110, "alias") for alias in spec.aliases)

    role_nouns = ROLE_NOUNS.get(spec.role, ())
    if spec.room:
        terms.extend((f"{spec.room}{noun}", 90, "room+role") for noun in role_nouns)
        terms.extend((f"{spec.room} {noun}", 90, "room+role") for noun in role_nouns)
        terms.append((f"{spec.room}{spec.role}", 70, "room+role"))
        terms.append((f"{spec.room} {spec.role}", 70, "room+role"))

    return tuple(terms)


def _device_hit(utterance: str, device: ResolvedDevice) -> _DeviceHit | None:
    text = utterance.casefold()
    compact_text = _compact(utterance)

    matches: list[tuple[int, str]] = []
    for term, weight, reason in _device_terms(device):
        if _contains(text, compact_text, term):
            matches.append((weight + len(_compact(term)), f"{reason}:{term}"))

    if not matches:
        return None

    score = max(score for score, _ in matches)
    reasons = tuple(reason for _, reason in sorted(matches, key=lambda item: (-item[0], item[1])))
    return _DeviceHit(device=device, score=score, reasons=reasons)


def _device_hits(utterance: str, resolver: DeviceResolver) -> list[_DeviceHit]:
    hits = [hit for device in resolver.all() if (hit := _device_hit(utterance, device)) is not None]
    return sorted(hits, key=lambda hit: (-hit.score, hit.device.spec.friendly_name, hit.device.local_id))


def _select_device(utterance: str, resolver: DeviceResolver) -> _DeviceHit:
    hits = _device_hits(utterance, resolver)
    if not hits:
        raise ReactiveMatchError(
            "unknown_device",
            "could not identify a managed device in the utterance",
        )

    top_score = hits[0].score
    top = [hit for hit in hits if hit.score == top_score]
    if len(top) > 1:
        raise ReactiveMatchError(
            "ambiguous_device",
            "utterance matches multiple devices with the same confidence",
            candidates=tuple(hit.device.spec.friendly_name for hit in top),
        )
    return hits[0]


def match_device_action(utterance: str, resolver: DeviceResolver) -> ReactiveMatch:
    action = _select_action(utterance)
    device = _select_device(utterance, resolver)

    supported_actions = effective_actions(device.device.spec)
    if action.action not in supported_actions:
        available = tuple(sorted(supported_actions))
        raise ReactiveMatchError(
            "unsupported_action",
            (
                f"device {device.device.spec.friendly_name!r} does not support "
                f"action {action.action!r}; available actions: {available}"
            ),
            candidates=available,
        )

    return ReactiveMatch(
        device=device.device,
        action=action.action,
        action_terms=action.terms,
        device_reasons=device.reasons,
    )
