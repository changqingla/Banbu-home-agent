from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import Request

from banbu.config.settings import Settings

from .feishu_adapter import IMAdapterError
from .types import IMAttachment, IncomingIMMessage, make_message_id


class WeixinBridgeAdapter:
    """Adapter for personal WeChat bridge connectors.

    Personal WeChat does not expose a clean official bot webhook surface. This
    adapter intentionally accepts a small bridge protocol so the connector that
    owns QR login / long polling can remain replaceable.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def verify_request(self, request: Request) -> None:
        expected = self._settings.im_weixin_bridge_token
        if not expected:
            return
        token = request.headers.get("x-banbu-im-token") or request.headers.get("authorization", "")
        if token.startswith("Bearer "):
            token = token[len("Bearer ") :]
        if token != expected:
            raise IMAdapterError("weixin bridge token mismatch")

    def parse_message(self, body: dict[str, Any]) -> IncomingIMMessage:
        conversation_id = self._first_text(body, "conversation_id", "chat_id", "from_user_id")
        if not conversation_id:
            raise IMAdapterError("weixin bridge message missing conversation_id")

        text = self._first_text(body, "text", "content", "utterance").strip()
        if not text:
            raise IMAdapterError("weixin bridge message has no text")

        user_id = self._first_text(body, "user_id", "from_user_id", default=conversation_id)
        message_id = self._first_text(body, "message_id", "msg_id", default=make_message_id("weixin"))
        ts = self._timestamp(body.get("timestamp") or body.get("create_time"))

        return IncomingIMMessage(
            platform="weixin",
            message_id=message_id,
            chat_id=conversation_id,
            user_id=f"weixin:{user_id}",
            user_display_name=self._first_text(body, "user_name", "display_name") or None,
            text=text,
            home_id=str(body.get("home_id") or self._settings.home_id),
            attachments=self._attachments(body.get("attachments")),
            timestamp=ts,
            raw=body,
        )

    async def send_text(self, message: IncomingIMMessage, text: str) -> str | None:
        reply_url = str(message.raw.get("response_url") or self._settings.im_weixin_reply_url or "").strip()
        if not reply_url:
            return None

        async with httpx.AsyncClient(
            timeout=self._settings.im_weixin_reply_timeout_seconds,
            trust_env=False,
        ) as client:
            resp = await client.post(
                reply_url,
                json={
                    "conversation_id": message.chat_id,
                    "message_id": message.message_id,
                    "text": text,
                },
            )
            resp.raise_for_status()
            return message.message_id

    def _first_text(self, body: dict[str, Any], *keys: str, default: str = "") -> str:
        for key in keys:
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    def _attachments(self, raw: Any) -> list[IMAttachment]:
        if not isinstance(raw, list):
            return []
        attachments: list[IMAttachment] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            attachments.append(
                IMAttachment(
                    kind=str(item.get("kind") or item.get("type") or "file"),
                    file_id=str(item.get("file_id") or "") or None,
                    file_name=str(item.get("file_name") or item.get("name") or "") or None,
                    url=str(item.get("url") or "") or None,
                    path=str(item.get("path") or "") or None,
                )
            )
        return attachments

    def _timestamp(self, value: Any) -> float:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return time.time()
        return raw / 1000 if raw > 10_000_000_000 else raw
