from __future__ import annotations

from .definition import ResolvedDevice


class DeviceResolver:
    """Bidirectional lookup between friendly_name / local_id / ieee_address."""

    def __init__(self, devices: list[ResolvedDevice]) -> None:
        self._by_name: dict[str, ResolvedDevice] = {d.spec.friendly_name: d for d in devices}
        self._by_local_id: dict[int, ResolvedDevice] = {d.local_id: d for d in devices}
        self._by_ieee: dict[str, ResolvedDevice] = {d.ieee_address: d for d in devices}
    def all(self) -> list[ResolvedDevice]:
        return list(self._by_name.values())

    def by_name(self, friendly_name: str) -> ResolvedDevice | None:
        return self._by_name.get(friendly_name)

    def by_local_id(self, local_id: int) -> ResolvedDevice | None:
        return self._by_local_id.get(local_id)

    def by_ieee(self, ieee_address: str) -> ResolvedDevice | None:
        return self._by_ieee.get(ieee_address)

    def resolve(self, key: str | int) -> ResolvedDevice | None:
        if isinstance(key, int):
            return self.by_local_id(key)
        return self.by_name(key) or self.by_ieee(key)
