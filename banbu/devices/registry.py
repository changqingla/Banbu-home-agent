"""Boot-time device registry.

Loads `config/devices.yaml`, reconciles each entry against the IoT platform's
`/api/v1/devices` listing and `/api/v1/exposes` capability spec, configures
emergency-keyword push for declared `care_fields`, and returns a DeviceResolver
for downstream lookups.

Validation failures abort the boot — partial state is never installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from banbu.adapters.iot_client import IoTClient

from .definition import DeviceSpec, DevicesFile, ResolvedDevice, effective_actions
from .resolver import DeviceResolver

log = logging.getLogger(__name__)


class RegistryError(RuntimeError):
    pass


def _load_yaml(path: Path) -> DevicesFile:
    if not path.exists():
        raise RegistryError(
            f"devices file not found: {path}. Copy banbu/config/devices.yaml.example "
            f"to {path} and edit to match your environment."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        raise RegistryError(f"devices file is empty: {path}")
    try:
        return DevicesFile.model_validate(raw)
    except ValidationError as e:
        raise RegistryError(f"devices file failed schema validation: {e}") from e


def _capabilities(exposes_doc: Any) -> set[str]:
    """Walk the IoT /exposes response and collect every leaf property name.

    The spec mixes flat entries (with a top-level `property`) and grouped
    entries (with `features: [...]`). We collect both.
    """
    caps: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            prop = node.get("property") or node.get("name")
            if isinstance(prop, str) and node.get("type") != "switch":
                caps.add(prop)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    if isinstance(exposes_doc, dict):
        walk(exposes_doc.get("exposes", exposes_doc))
    else:
        walk(exposes_doc)
    return caps


async def build_registry(
    client: IoTClient,
    devices_path: Path,
    *,
    configure_emergency: bool = True,
    strict: bool = False,
) -> DeviceResolver:
    spec_file = _load_yaml(devices_path)
    declared = spec_file.devices

    iot_devices = await client.list_devices()
    by_name: dict[str, dict[str, Any]] = {d["friendly_name"]: d for d in iot_devices}

    missing = [s.friendly_name for s in declared if not s.virtual and s.friendly_name not in by_name]
    if missing:
        message = (
            "devices.yaml references friendly_names not found on IoT platform: "
            f"{missing}"
        )
        if strict:
            raise RegistryError(message)
        log.warning("%s; skipping because registry strict mode is disabled", message)

    resolved: list[ResolvedDevice] = []
    errors: list[str] = []
    used_virtual_ids: set[int] = set()

    for spec in declared:
        if spec.virtual:
            if spec.local_id is None:
                errors.append(f"{spec.friendly_name}: virtual devices must set local_id")
                continue
            if spec.local_id >= 0:
                errors.append(f"{spec.friendly_name}: virtual local_id should be negative, got {spec.local_id}")
                continue
            if spec.local_id in used_virtual_ids:
                errors.append(f"{spec.friendly_name}: duplicate virtual local_id {spec.local_id}")
                continue
            used_virtual_ids.add(spec.local_id)
            caps = set(spec.capabilities or spec.care_fields)
            missing_caps = [f for f in spec.care_fields if f not in caps]
            if missing_caps:
                errors.append(
                    f"{spec.friendly_name}: care_fields {missing_caps} not in virtual capabilities {sorted(caps)}"
                )
                continue
            resolved.append(
                ResolvedDevice(
                    spec=spec,
                    local_id=spec.local_id,
                    ieee_address=spec.ieee_address or f"virtual:{spec.friendly_name}",
                    model=spec.model or "virtual",
                    capabilities=caps,
                )
            )
            continue

        if spec.friendly_name not in by_name:
            continue
        iot_dev = by_name[spec.friendly_name]
        local_id = int(iot_dev["local_id"])
        ieee = str(iot_dev["ieee_address"])

        try:
            exposes_doc = await client.get_exposes(local_id)
        except Exception as e:
            errors.append(f"{spec.friendly_name}: failed to fetch /exposes ({e})")
            continue

        caps = _capabilities(exposes_doc)

        unknown_care = [f for f in spec.care_fields if f not in caps]
        if unknown_care:
            errors.append(
                f"{spec.friendly_name}: care_fields {unknown_care} not in capabilities {sorted(caps)}"
            )

        for action_name, payload in effective_actions(spec).items():
            unknown_action = [k for k in payload if k not in caps]
            if unknown_action:
                errors.append(
                    f"{spec.friendly_name}: action '{action_name}' uses unknown fields {unknown_action}"
                )

        resolved.append(
            ResolvedDevice(
                spec=spec,
                local_id=local_id,
                ieee_address=ieee,
                model=str(iot_dev.get("model", "")),
                capabilities=caps,
            )
        )

    if errors:
        joined = "\n  - ".join(errors)
        raise RegistryError(f"devices.yaml validation errors:\n  - {joined}")

    if configure_emergency:
        for d in resolved:
            if d.spec.virtual:
                log.info(
                    "report-config skipped for virtual device %s (local_id=%d)",
                    d.spec.friendly_name,
                    d.local_id,
                )
                continue
            if not d.spec.care_fields:
                continue
            try:
                await client.set_report_config(d.local_id, emergency_keywords=d.spec.care_fields)
                log.info(
                    "report-config %s (local_id=%d) emergency_keywords=%s",
                    d.spec.friendly_name,
                    d.local_id,
                    d.spec.care_fields,
                )
            except Exception as e:
                errors.append(f"{d.spec.friendly_name}: report-config failed ({e})")

        if errors:
            joined = "\n  - ".join(errors)
            raise RegistryError(f"failed to configure emergency keywords:\n  - {joined}")

    return DeviceResolver(resolved, skipped_missing_devices=missing)


async def load_specs(devices_path: Path) -> list[DeviceSpec]:
    """Read devices.yaml without touching the network. Used by tests / probes."""
    return _load_yaml(devices_path).devices
