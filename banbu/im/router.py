from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from banbu.config.settings import Settings
from banbu.reactive.runner import ReactiveRunner, ReactiveRunResult, result_payload
from banbu.turn.builder import from_reactive
from banbu.turn.scheduler import TurnScheduler, reactive_key

from .feishu_adapter import IMAdapterError
from .feishu_service import FeishuSDKService
from .reply import render_reactive_reply
from .types import IncomingIMMessage
from .weixin_adapter import WeixinBridgeAdapter

log = logging.getLogger(__name__)


def make_router(
    *,
    settings: Settings,
    runner: ReactiveRunner,
    scheduler: TurnScheduler,
) -> APIRouter:
    router = APIRouter()
    weixin = WeixinBridgeAdapter(settings)

    async def run_message(message: IncomingIMMessage) -> ReactiveRunResult:
        turn = from_reactive(
            message.text,
            home_id=message.home_id,
            user_id=message.user_id,
            source=message.source,
        )
        holder: dict[str, ReactiveRunResult] = {}

        async def job() -> None:
            holder["result"] = await runner.run(turn)

        await scheduler.run_serialized(reactive_key(turn.home_id, turn.user_id or "unknown"), job)
        return holder["result"]

    def disabled(platform: str) -> dict[str, Any]:
        return {"ok": False, "ignored": True, "reason": f"{platform}_disabled"}

    async def try_reply(send) -> tuple[str | None, str | None]:
        try:
            return await send(), None
        except Exception as e:
            log.warning("im reply failed: %s", e)
            return None, str(e)

    @router.post(settings.im_weixin_path)
    async def weixin_messages(request: Request) -> dict[str, Any]:
        if not settings.im_enabled or not settings.im_weixin_enabled:
            return disabled("weixin")
        try:
            weixin.verify_request(request)
            body = await request.json()
        except IMAdapterError as e:
            log.warning("weixin: unauthorized bridge request: %s", e)
            return {"ok": False, "ignored": True, "reason": str(e)}
        except Exception:
            return {"ok": False, "ignored": True, "reason": "invalid_json"}

        if not isinstance(body, dict):
            return {"ok": False, "ignored": True, "reason": "invalid_body"}

        try:
            message = weixin.parse_message(body)
        except IMAdapterError as e:
            log.warning("weixin: ignored bridge message: %s", e)
            return {"ok": False, "ignored": True, "reason": str(e)}

        result = await run_message(message)
        reply = render_reactive_reply(result)
        reply_message_id, reply_error = await try_reply(lambda: weixin.send_text(message, reply))
        return {
            "ok": result.ok,
            "platform": message.platform,
            "message_id": message.message_id,
            "reply": reply,
            "reply_message_id": reply_message_id,
            "reply_error": reply_error,
            "result": result_payload(result),
        }

    return router


def make_feishu_sdk_service(
    *,
    settings: Settings,
    runner: ReactiveRunner,
    scheduler: TurnScheduler,
) -> FeishuSDKService:
    async def run_message(message: IncomingIMMessage) -> ReactiveRunResult:
        turn = from_reactive(
            message.text,
            home_id=message.home_id,
            user_id=message.user_id,
            source=message.source,
        )
        holder: dict[str, ReactiveRunResult] = {}

        async def job() -> None:
            holder["result"] = await runner.run(turn)

        await scheduler.run_serialized(reactive_key(turn.home_id, turn.user_id or "unknown"), job)
        return holder["result"]

    return FeishuSDKService(settings=settings, run_message=run_message)
