from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

from banbu.config.settings import Settings

from .types import IMAttachment, IncomingIMMessage, make_message_id


class IMAdapterError(ValueError):
    pass


class FeishuAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: lark.Client | None = None

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

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(self._receive_id_type(chat_id))
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .uuid(make_message_id("feishu_reply"))
                .build()
            )
            .build()
        )
        response = await self._send_message(request)
        if not response.success():
            raise IMAdapterError(
                f"feishu send message failed: code={response.code} msg={response.msg}"
            )
        if response.data is None:
            return ""
        return str(response.data.message_id or "")

    async def _send_message(self, request: CreateMessageRequest) -> Any:
        import asyncio

        return await asyncio.to_thread(self._send_message_sync, request)

    def _send_message_sync(self, request: CreateMessageRequest) -> Any:
        with _without_invalid_socks_proxy():
            return self._sdk_client().im.v1.message.create(request)

    def parse_sdk_message(self, event: Any) -> IncomingIMMessage:
        payload = {
            "header": self._sdk_header(event),
            "event": {
                "sender": self._sdk_sender(event),
                "message": self._sdk_message(event),
            },
        }
        return self.parse_event(payload)

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

    def _sdk_client(self) -> lark.Client:
        if self._client is None:
            self._client = (
                lark.Client.builder()
                .app_id(self._settings.im_feishu_app_id)
                .app_secret(self._settings.im_feishu_app_secret)
                .domain(self._settings.im_feishu_api_base_url.rstrip("/"))
                .timeout(float(self._settings.iot_timeout_seconds))
                .build()
            )
        return self._client

    def _receive_id_type(self, receive_id: str) -> str:
        if receive_id.startswith("oc_"):
            return "chat_id"
        if receive_id.startswith("ou_"):
            return "open_id"
        if receive_id.startswith("on_"):
            return "union_id"
        return "user_id"

    def _sdk_header(self, event: Any) -> dict[str, Any]:
        header = getattr(event, "header", None)
        if header is None:
            return {}
        return {
            "event_id": getattr(header, "event_id", None),
            "token": getattr(header, "token", None),
            "create_time": getattr(header, "create_time", None),
        }

    def _sdk_sender(self, event: Any) -> dict[str, Any]:
        sender = getattr(getattr(event, "event", None), "sender", None)
        sender_id = getattr(sender, "sender_id", None)
        return {
            "sender_id": {
                "user_id": getattr(sender_id, "user_id", None),
                "open_id": getattr(sender_id, "open_id", None),
                "union_id": getattr(sender_id, "union_id", None),
            },
            "sender_type": getattr(sender, "sender_type", None),
            "tenant_key": getattr(sender, "tenant_key", None),
        }

    def _sdk_message(self, event: Any) -> dict[str, Any]:
        message = getattr(getattr(event, "event", None), "message", None)
        return {
            "message_id": getattr(message, "message_id", None),
            "chat_id": getattr(message, "chat_id", None),
            "message_type": getattr(message, "message_type", None),
            "content": getattr(message, "content", None),
            "create_time": getattr(message, "create_time", None),
        }


@contextmanager
def _without_invalid_socks_proxy():
    keys = ["ALL_PROXY", "all_proxy"]
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            value = os.environ.get(key, "")
            if value.lower().startswith("socks://"):
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
