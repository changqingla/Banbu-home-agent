from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from banbu.devices.registry import RegistryError, build_registry
from banbu.scenes.loader import SceneLoadError, load_scenes


class InMemoryIoTClient:
    def __init__(self) -> None:
        self.exposes_requested: list[int] = []

    async def list_devices(self) -> list[dict[str, Any]]:
        return [
            {
                "friendly_name": "present_switch",
                "local_id": 12,
                "ieee_address": "0xpresent",
                "model": "TS011F",
            }
        ]

    async def get_exposes(self, local_id: int) -> dict[str, Any]:
        self.exposes_requested.append(local_id)
        return {"exposes": [{"property": "state", "type": "binary"}]}

    async def set_report_config(
        self,
        local_id: int,
        *,
        emergency_keywords: list[str] | None = None,
        care_keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        return {"ok": True}


def _devices_yaml(path: Path) -> Path:
    path.write_text(
        """
devices:
  - friendly_name: present_switch
    role: light_switch
    care_fields: [state]
    actions:
      turn_on: { state: "ON" }
      turn_off: { state: "OFF" }

  - friendly_name: missing_sensor
    role: door_sensor
    care_fields: [contact]
""".strip(),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_strict_registry_raises_for_missing_declared_device(tmp_path: Path) -> None:
    client = InMemoryIoTClient()

    with pytest.raises(RegistryError, match="missing_sensor"):
        await build_registry(
            client,
            _devices_yaml(tmp_path / "devices.yaml"),
            configure_emergency=False,
            strict=True,
        )

    assert client.exposes_requested == []


@pytest.mark.asyncio
async def test_non_strict_registry_skips_missing_declared_device(tmp_path: Path) -> None:
    client = InMemoryIoTClient()

    resolver = await build_registry(
        client,
        _devices_yaml(tmp_path / "devices.yaml"),
        configure_emergency=False,
        strict=False,
    )

    assert resolver.by_name("present_switch") is not None
    assert resolver.by_name("missing_sensor") is None
    assert resolver.skipped_missing_devices == ["missing_sensor"]
    assert client.exposes_requested == [12]


@pytest.mark.asyncio
async def test_scene_loader_rejects_scene_referencing_skipped_device(tmp_path: Path) -> None:
    resolver = await build_registry(
        InMemoryIoTClient(),
        _devices_yaml(tmp_path / "devices.yaml"),
        configure_emergency=False,
        strict=False,
    )
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    (scenes_dir / "missing_scene.yaml").write_text(
        """
scene_id: missing_device_scene
name: Missing device scene
kind: sequential
trigger:
  steps:
    - device: missing_sensor
      field: payload.contact
      transition: "true->false"
context_devices:
  trigger: [missing_sensor]
  context_only: []
intent: Should not load
actions_hint: []
policy:
  cooldown_seconds: 60
  inflight_seconds: 30
  priority: 5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(SceneLoadError, match="unknown device 'missing_sensor'"):
        load_scenes(scenes_dir, resolver)
