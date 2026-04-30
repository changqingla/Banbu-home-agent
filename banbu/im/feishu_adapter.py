from __future__ import annotations

import json
import time
from typing import Any

import httpx

from banbu.config.settings import Settings

from .types import IMAttachment, IncomingIMMessage, make_message_id


class IMAdapterError(ValueError):
    pass


class FeishuAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tenant_token: str | None = None
        self._tenant_token_expires_at = 0.0

    def verify_url_challenge(self, body: dict[str, Any]) -> dict[str, str] | None:
        challenge = body.get("challenge")
        if not isinstance(challenge, str):
            return None
        self._verify_token(body.get("token") or body.get("header", {}).get("token"))
        return {"challenge": challenge}

    def parse_event(self, body: dict[str, Any]) -> IncomingIMMessage:
        header = body.get("header") if isinstance(body.get("header"), dict) else {}
        self._verify_token(header.get("token") or body.get("token"))

        event = body.get("event")
        if not isinstance(event, dict):
            raise IMAdapterError("feishu event body missing event object")

        message = event.get("message")
        sender = event.get("sender")
        if not isinstance(message, dict) or not isinstance(sender, dict):
            raise IMAdapterError("feishu event missing message or sender")

        message_type = str(message.get("message_type") or "")
        content = self._parse_content(message.get("content"))
        text = self._extract_text(message_type, content).strip()
        if not text:
            raise IMAdapterError("feishu message has no supported text content")

        sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
        user_id = (
            sender_id.get("user_id")
            or sender_id.get("open_id")
            or sender_id.get("union_id")
            or "feishu_user"
        )
        chat_id = str(message.get("chat_id") or user_id)
        message_id = str(message.get("message_id") or header.get("event_id") or make_message_id("feishu"))
        ts = self._timestamp(message.get("create_time") or header.get("create_time"))

        return IncomingIMMessage(
            platform="feishu",
            message_id=message_id,
            chat_id=chat_id,
            user_id=f"feishu:{user_id}",
            text=text,
            home_id=self._settings.home_id,
            attachments=self._attachments(message_type, content),
            timestamp=ts,
            raw=body,
        )

    async def send_text(self, chat_id: str, text: str) -> str | None:
        if not self._settings.im_feishu_reply_enabled:
            return None
        if not self._settings.im_feishu_app_id or not self._settings.im_feishu_app_secret:
            raise IMAdapterError("feishu reply requires app_id and app_secret")

        token = await self._tenant_access_token()
        base = self._settings.im_feishu_api_base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=self._settings.iot_timeout_seconds, trust_env=False) as client:
            resp = await client.post(
                f"{base}/open-apis/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text}, ensure_ascii=False),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("data", {}).get("message_id") or "")

    def _verify_token(self, token: Any) -> None:
        expected = self._settings.im_feishu_verification_token
        if expected and token != expected:
            raise IMAdapterError("feishu verification token mismatch")

    def _parse_content(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise IMAdapterError(f"invalid feishu message content JSON: {e}") from e
            if isinstance(data, dict):
                return data
        return {}

    def _extract_text(self, message_type: str, content: dict[str, Any]) -> str:
        if message_type == "text":
            return str(content.get("text") or "")
        if message_type == "post":
            return self._extract_post_text(content)
        return ""

    def _extract_post_text(self, content: dict[str, Any]) -> str:
        post = content.get("post")
        if not isinstance(post, dict):
            return ""
        lines: list[str] = []
        for localized in post.values():
            if not isinstance(localized, dict):
                continue
            for line in localized.get("content") or []:
                if not isinstance(line, list):
                    continue
                parts = [
                    str(item.get("text") or "")
                    for item in line
                    if isinstance(item, dict) and item.get("tag") == "text"
                ]
                if parts:
                    lines.append("".join(parts))
        return "\n".join(lines)

    def _attachments(self, message_type: str, content: dict[str, Any]) -> list[IMAttachment]:
        if message_type == "image":
            return [IMAttachment(kind="image", file_id=str(content.get("image_key") or ""))]
        if message_type == "file":
            return [
                IMAttachment(
                    kind="file",
                    file_id=str(content.get("file_key") or ""),
                    file_name=str(content.get("file_name") or ""),
                )
            ]
        return []

    def _timestamp(self, value: Any) -> float:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return time.time()
        return raw / 1000 if raw > 10_000_000_000 else raw

    async def _tenant_access_token(self) -> str:
        now = time.time()
        if self._tenant_token and now < self._tenant_token_expires_at:
            return self._tenant_token

        base = self._settings.im_feishu_api_base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=self._settings.iot_timeout_seconds, trust_env=False) as client:
            resp = await client.post(
                f"{base}/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self._settings.im_feishu_app_id,
                    "app_secret": self._settings.im_feishu_app_secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        token = data.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise IMAdapterError("feishu tenant_access_token response missing token")
        expire = data.get("expire")
        ttl = float(expire) if isinstance(expire, (int, float)) else 3600.0
        self._tenant_token = token
        self._tenant_token_expires_at = now + max(ttl - 60, 60)
        return token
