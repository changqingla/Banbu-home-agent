from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Scene

from .matcher import ROLE_NOUNS

SceneMatchErrorKind = Literal["no_scene_match", "ambiguous_scene"]

_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "if",
    "is",
    "of",
    "or",
    "the",
    "then",
    "to",
    "when",
    "with",
}


@dataclass(frozen=True)
class SceneMatchCandidate:
    scene: Scene
    score: int
    reasons: tuple[str, ...]


class SceneMatchError(ValueError):
    def __init__(
        self,
        kind: SceneMatchErrorKind,
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


def _features(text: str) -> set[str]:
    features: set[str] = set()
    for word in re.findall(r"[a-zA-Z0-9_]+", text.casefold()):
        for part in word.split("_"):
            if len(part) > 1 and part not in _EN_STOPWORDS:
                features.add(part)
    for cjk in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(cjk) > 1:
            features.add(cjk)
        for width in (2, 3):
            if len(cjk) >= width:
                features.update(cjk[i : i + width] for i in range(len(cjk) - width + 1))
    return features


def _score_text_field(
    *,
    utterance: str,
    compact_utterance: str,
    utterance_features: set[str],
    value: str,
    label: str,
    direct_weight: int,
    overlap_weight: int,
) -> list[tuple[int, str]]:
    if not value:
        return []

    scores: list[tuple[int, str]] = []
    compact_value = _compact(value)
    if compact_value and (compact_value in compact_utterance or compact_utterance in compact_value):
        scores.append((direct_weight + len(compact_value), f"{label}:direct:{value}"))

    overlap = sorted(utterance_features & _features(value))
    if len(overlap) >= 2:
        scores.append((overlap_weight + len(overlap) * 6, f"{label}:overlap:{','.join(overlap[:6])}"))

    return scores


def _score_device_metadata(
    *,
    scene: Scene,
    resolver: DeviceResolver,
    utterance: str,
    compact_utterance: str,
) -> list[tuple[int, str]]:
    scores: list[tuple[int, str]] = []
    for device_name in sorted(scene.all_referenced_devices()):
        device = resolver.by_name(device_name)
        if device is None:
            continue

        spec = device.spec
        terms: list[tuple[str, int, str]] = [(spec.friendly_name, 75, "device:friendly_name")]
        terms.extend((alias, 85, "device:alias") for alias in spec.aliases)

        role_nouns = ROLE_NOUNS.get(spec.role, ())
        if spec.room:
            terms.append((spec.room, 30, "device:room"))
            terms.extend((f"{spec.room}{noun}", 75, "device:room+role") for noun in role_nouns)
            terms.extend((f"{spec.room} {noun}", 75, "device:room+role") for noun in role_nouns)
            if _contains(utterance, compact_utterance, spec.room):
                for noun in role_nouns:
                    if _contains(utterance, compact_utterance, noun):
                        terms.append((f"{spec.room}+{noun}", 70, "device:room+role_parts"))

        for term, weight, label in terms:
            if "+" in term and label.endswith("_parts"):
                scores.append((weight + len(_compact(term)), f"{label}:{spec.friendly_name}:{term}"))
            elif _contains(utterance, compact_utterance, term):
                scores.append((weight + len(_compact(term)), f"{label}:{spec.friendly_name}:{term}"))

    return scores


def score_scene(utterance: str, scene: Scene, resolver: DeviceResolver) -> SceneMatchCandidate | None:
    compact_utterance = _compact(utterance)
    utterance_features = _features(utterance)

    scores: list[tuple[int, str]] = []
    scores.extend(
        _score_text_field(
            utterance=utterance,
            compact_utterance=compact_utterance,
            utterance_features=utterance_features,
            value=scene.scene_id,
            label="scene_id",
            direct_weight=130,
            overlap_weight=70,
        )
    )
    scores.extend(
        _score_text_field(
            utterance=utterance,
            compact_utterance=compact_utterance,
            utterance_features=utterance_features,
            value=scene.name,
            label="name",
            direct_weight=115,
            overlap_weight=65,
        )
    )
    scores.extend(
        _score_text_field(
            utterance=utterance,
            compact_utterance=compact_utterance,
            utterance_features=utterance_features,
            value=scene.intent,
            label="intent",
            direct_weight=100,
            overlap_weight=55,
        )
    )
    scores.extend(
        _score_device_metadata(
            scene=scene,
            resolver=resolver,
            utterance=utterance,
            compact_utterance=compact_utterance,
        )
    )

    if not scores:
        return None

    score = max(score for score, _ in scores)
    reasons = tuple(reason for _, reason in sorted(scores, key=lambda item: (-item[0], item[1])))
    return SceneMatchCandidate(scene=scene, score=score, reasons=reasons)


def rank_scene_candidates(
    utterance: str,
    scenes: list[Scene],
    resolver: DeviceResolver,
) -> list[SceneMatchCandidate]:
    candidates = [candidate for scene in scenes if (candidate := score_scene(utterance, scene, resolver))]
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.scene.scene_id))


def select_scene_match(
    utterance: str,
    scenes: list[Scene],
    resolver: DeviceResolver,
) -> SceneMatchCandidate:
    candidates = rank_scene_candidates(utterance, scenes, resolver)
    if not candidates:
        raise SceneMatchError(
            "no_scene_match",
            "could not match the utterance to a configured scene",
        )

    top_score = candidates[0].score
    top = [candidate for candidate in candidates if candidate.score == top_score]
    if len(top) > 1:
        raise SceneMatchError(
            "ambiguous_scene",
            "utterance matches multiple scenes with the same confidence",
            candidates=tuple(candidate.scene.scene_id for candidate in top),
        )
    return candidates[0]
