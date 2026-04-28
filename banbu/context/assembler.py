"""Pack a SelectedContext into a stable list of context blocks (plan §9.4).

Each block is a self-contained string. Order is deliberate:
  static-frame blocks first (more cache-friendly across requests),
  per-scene definition next, then per-turn dynamic facts last.
"""
from __future__ import annotations

import json

from banbu.agent.prompts import SYSTEM_POLICY
from banbu.devices.definition import effective_actions

from .selector import SelectedContext


def _device_block(dev, snap) -> str:
    actions = sorted(effective_actions(dev.spec))
    payload = json.dumps(snap.payload if snap else {}, ensure_ascii=False, sort_keys=True)
    return (
        f"[device:{dev.spec.friendly_name}] local_id={dev.local_id} "
        f"role={dev.spec.role} "
        f"actions={actions or 'none'} "
        f"snapshot={payload}"
    )


def _scene_block(scene, name_to_local_id: dict) -> str:
    steps = "; ".join(
        f"step{i+1}: {s.device}.{s.field} {s.transition}"
        + (f" within={s.within_seconds}s" if s.within_seconds else "")
        for i, s in enumerate(scene.trigger.steps)
    )
    pres = "; ".join(
        f"{p.device}.{p.field} {p.op} {p.value!r} (on_missing={p.on_missing})"
        for p in scene.preconditions
    )
    if pres:
        pres = f"[pre-verified] {pres}"
    hints = []
    for h in scene.actions_hint:
        local_id = name_to_local_id.get(h.args.get("device", ""), "?")
        action = h.args.get("action", "?")
        hints.append(f"local_id={local_id} action={action}")
    hint = "; ".join(hints)
    return (
        f"[scene:{scene.scene_id}] name={scene.name!r} kind={scene.kind} "
        f"intent={scene.intent!r}\n"
        f"  trigger: {steps}\n"
        f"  preconditions: {pres or '(none)'}\n"
        f"  actions_hint: {hint or '(none)'}"
    )


def _trigger_facts_block(ctx: SelectedContext) -> str:
    trg = ctx.turn.trigger
    if trg is None:
        return "[trigger] (no proactive trigger)"
    summary = "\n  ".join(trg.source_event_summaries) or "(no summaries)"
    return (
        f"[trigger] id={trg.trigger_id} scene={trg.scene_id} home={trg.home_id}\n"
        f"  triggered_at={trg.triggered_at:.0f}\n"
        f"  source events:\n  {summary}"
    )


def assemble_blocks(ctx: SelectedContext) -> list[str]:
    blocks: list[str] = []

    name_to_local_id = {dev.spec.friendly_name: dev.local_id for dev in ctx.devices}
    blocks.append(f"[system policy]\n{SYSTEM_POLICY}")
    blocks.append(_scene_block(ctx.scene, name_to_local_id))
    blocks.append(_trigger_facts_block(ctx))

    for dev in ctx.devices:
        snap = ctx.snapshots.get(dev.spec.friendly_name)
        blocks.append(_device_block(dev, snap))

    return blocks
