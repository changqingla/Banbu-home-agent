from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as FeishuWSClient

from banbu.config.settings import Settings
from banbu.reactive.runner import ReactiveRunResult
from banbu.turn.scheduler import reactive_key

from .feishu_adapter import FeishuAdapter, IMAdapterError
from .reply import render_reactive_reply
from .types import IncomingIMMessage

log = logging.getLogger(__name__)

MessageRunner = Callable[[IncomingIMMessage], Awaitable[ReactiveRunResult]]


class FeishuSDKService:
    """Official Feishu SDK WebSocket receiver for IM reactive turns."""

    def __init__(
        self,
        *,
        settings: Settings,
        run_message: MessageRunner,
    ) -> None:
        self._settings = settings
        self._adapter = FeishuAdapter(settings)
        self._run_message = run_message
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws_client: FeishuWSClient | None = None
        self._stop_event = threading.Event()
        self._processed_events: dict[str, float] = {}

    def start(self) -> None:
        if not self._enabled():
            log.info("feishu SDK WebSocket service disabled")
            return
        if not self._settings.im_feishu_app_id or not self._settings.im_feishu_app_secret:
            raise RuntimeError("feishu SDK WebSocket requires app_id and app_secret")
        if self._thread is not None and self._thread.is_alive():
            return

        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_ws_client,
            name="banbu-feishu-sdk-ws",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "feishu SDK WebSocket service starting app_id=%s token_configured=%s encrypt_key_configured=%s",
            self._settings.im_feishu_app_id,
            bool(self._settings.im_feishu_verification_token),
            bool(self._settings.im_feishu_encrypt_key),
        )

    async def stop(self) -> None:
        self._stop_event.set()
        client = self._ws_client
        if client is not None:
            with suppress(Exception):
                await asyncio.to_thread(self._stop_ws_client, client)
        thread = self._thread
        if thread is None:
            return
        await asyncio.to_thread(thread.join, 5)
        self._thread = None
        self._ws_client = None

    def _stop_ws_client(self, client: FeishuWSClient) -> None:
        import lark_oapi.ws.client as ws_client_module

        with suppress(Exception):
            ws_client_module.loop.run_until_complete(client._disconnect())

    def _enabled(self) -> bool:
        return (
            self._settings.im_enabled
            and self._settings.im_feishu_enabled
        )

    def _run_ws_client(self) -> None:
        import lark_oapi.ws.client as ws_client_module

        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)
        ws_client_module.loop = ws_loop
        log.info("feishu SDK WebSocket event loop initialized in thread=%s", threading.current_thread().name)
        while not self._stop_event.is_set():
            try:
                handler = (
                    EventDispatcherHandler.builder(
                        self._settings.im_feishu_encrypt_key,
                        self._settings.im_feishu_verification_token,
                        lark.LogLevel.INFO,
                    )
                    .register_p2_im_message_receive_v1(self._on_sdk_message)
                    .register_p2_im_message_message_read_v1(self._on_sdk_ignored_event)
                    .build()
                )
                self._ws_client = FeishuWSClient(
                    self._settings.im_feishu_app_id,
                    self._settings.im_feishu_app_secret,
                    log_level=lark.LogLevel.INFO,
                    event_handler=handler,
                    domain=self._settings.im_feishu_api_base_url.rstrip("/"),
                )
                log.info("feishu SDK WebSocket client connecting")
                self._ws_client.start()
            except Exception as e:
                if self._stop_event.is_set():
                    return
                log.warning("feishu SDK WebSocket client stopped unexpectedly: %s", e)
                time.sleep(5)

    def _on_sdk_message(self, event: object) -> None:
        if self._loop is None or self._stop_event.is_set():
            return
        log.info("feishu SDK: received im.message.receive_v1 event")
        try:
            message = self._adapter.parse_sdk_message(event)
        except IMAdapterError as e:
            log.warning("feishu SDK: ignored inbound event: %s", e)
            return
        if self._already_processed(message.message_id):
            log.info("feishu SDK: ignored duplicate message_id=%s", message.message_id)
            return
        log.info(
            "feishu SDK: scheduling reactive turn message_id=%s chat_id=%s user_id=%s text=%r",
            message.message_id,
            message.chat_id,
            message.user_id,
            message.text,
        )
        asyncio.run_coroutine_threadsafe(self._handle_message(message), self._loop)

    def _on_sdk_ignored_event(self, event: object) -> None:
        event_type = getattr(getattr(event, "header", None), "event_type", "unknown")
        log.debug("feishu SDK: ignored event type=%s", event_type)

    def _already_processed(self, message_id: str) -> bool:
        now = time.time()
        ttl = max(self._settings.im_feishu_event_dedupe_ttl_seconds, 1)
        expired = [
            key for key, timestamp in self._processed_events.items()
            if now - timestamp > ttl
        ]
        for key in expired:
            self._processed_events.pop(key, None)
        if message_id in self._processed_events:
            return True
        self._processed_events[message_id] = now
        return False

    async def _handle_message(self, message: IncomingIMMessage) -> None:
        try:
            result = await self._run_message(message)
            reply = render_reactive_reply(result)
            try:
                reply_message_id = await self._adapter.send_text(message.chat_id, reply)
                log.info(
                    "feishu SDK: sent reply original_message_id=%s reply_message_id=%s",
                    message.message_id,
                    reply_message_id,
                )
            except Exception:
                log.exception("feishu SDK: failed to send reply message_id=%s", message.message_id)
        except Exception:
            log.exception(
                "feishu SDK: reactive turn failed key=%s message_id=%s",
                reactive_key(message.home_id, message.user_id),
                message.message_id,
            )
