"""Async wrapper around the IoT platform REST API.

Endpoints used (all under /api/v1):
  GET  /devices              list of devices
  GET  /devices/info         single device latest payload
  GET  /devices/allinfo      all devices latest payload
  GET  /exposes              capability spec
  GET  /devices/history      recent payload history
  POST /devices/set          control a device
  POST /devices/report-config configure care/emergency keywords
"""
from __future__ import annotations

from typing import Any

import httpx

from banbu.config.settings import Settings, get_settings


class IoTError(RuntimeError):
    pass


class IoTClient:
    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.iot_base_url,
            timeout=self._settings.iot_timeout_seconds,
            trust_env=False,
        )

    async def __aenter__(self) -> "IoTClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, **params: Any) -> Any:
        resp = await self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        if resp.status_code >= 400:
            raise IoTError(f"GET {path} -> {resp.status_code}: {resp.text}")
        return resp.json()

    async def _post(self, path: str, *, params: dict[str, Any] | None = None, json: Any = None) -> Any:
        resp = await self._client.post(path, params=params or {}, json=json)
        if resp.status_code >= 400:
            raise IoTError(f"POST {path} -> {resp.status_code}: {resp.text}")
        return resp.json()

    # ── reads ────────────────────────────────────────────────────────
    async def list_devices(self) -> list[dict[str, Any]]:
        return await self._get("/api/v1/devices")

    async def list_online_devices(self, max_age_seconds: int = 300) -> list[dict[str, Any]]:
        return await self._get("/api/v1/devices/online", max_age_seconds=max_age_seconds)

    async def get_device(self, local_id: int) -> dict[str, Any]:
        return await self._get("/api/v1/devices/one", local_id=local_id)

    async def get_exposes(self, local_id: int) -> dict[str, Any]:
        return await self._get("/api/v1/exposes", local_id=local_id)

    async def get_info(self, local_id: int) -> dict[str, Any]:
        return await self._get("/api/v1/devices/info", local_id=local_id)

    async def get_allinfo(self) -> list[dict[str, Any]]:
        return await self._get("/api/v1/devices/allinfo")

    async def get_history(self, local_id: int, limit: int = 20) -> list[dict[str, Any]]:
        return await self._get("/api/v1/devices/history", local_id=local_id, limit=limit)

    async def get_report_config(self, local_id: int) -> dict[str, Any]:
        return await self._get("/api/v1/devices/report-config", local_id=local_id)

    # ── writes ───────────────────────────────────────────────────────
    async def control(self, local_id: int, payload: dict[str, Any]) -> Any:
        return await self._post("/api/v1/devices/set", params={"local_id": local_id}, json={"payload": payload})

    async def set_report_config(
        self,
        local_id: int,
        *,
        emergency_keywords: list[str] | None = None,
        care_keywords: list[str] | None = None,
    ) -> Any:
        body: dict[str, Any] = {}
        if emergency_keywords is not None:
            body["emergency_keywords"] = emergency_keywords
        if care_keywords is not None:
            body["care_keywords"] = care_keywords
        return await self._post(
            "/api/v1/devices/report-config",
            params={"local_id": local_id},
            json=body,
        )
